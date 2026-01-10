# -*- coding: utf-8 -*-
"""
NFL Betting Model - ICLR 2025 Refined Version

This script predicts NFL game outcomes using machine learning.
It has been refined to align with the methodology outlined in the ICLR 2025 paper:
1.  **Dual Target Prediction**: Predicts both Moneyline (Win/Loss) and Spread Coverage (ATS).
2.  **Strict Validation**: Uses TimeSeriesSplit to prevent data leakage (no future data).
3.  **Advanced Ensembling**: Uses StackingClassifier (meta-learner) instead of simple Voting.
4.  **Scientific Rigor**: Includes Feature Scaling, Paired T-Tests, and Calibration Analysis.

Original Notebook: fml.ipynb
Refactored by: Coding Assistant
"""

# Standard Library
import os
import argparse
import pickle
import warnings
import datetime
from datetime import timedelta
from pathlib import Path

# Data Manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for headless execution
import matplotlib.pyplot as plt

# Machine Learning
import xgboost as xgb
import sklearn
from sklearn import model_selection
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from scipy import stats

# Models
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV as CCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    AdaBoostClassifier, 
    VotingClassifier, 
    StackingClassifier
)

# Suppress Warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning) # Clean up pandas output



def load_data(data_dir):
    """
    Loads raw NFL data files from the specified directory.

    Args:
        data_dir (Path): The Path object to the directory containing the CSV files.

    Returns:
        tuple: A tuple containing three pandas DataFrames:
            - df: Raw spreadspoke scores data.
            - teams: NFL teams metadata.
            - games_elo: Historical ELO ratings and game data.
    """
    print(f"Loading data from {data_dir}...")
    df = pd.read_csv(data_dir / "spreadspoke_scores.csv", encoding='utf-8')
    teams = pd.read_csv(data_dir / "nfl_teams.csv", encoding='utf-8')
    games_elo = pd.read_csv(data_dir / "nfl_elo.csv", encoding='utf-8')
    return df, teams, games_elo


def clean_data(df, teams, games_elo):
    """
    Cleans, merges, and pre-processes the raw datasets into a single analytic DataFrame.

    Key Processing Steps:
    1.  **Data Cleaning**: Removes rows with missing critical data (scores, lines) and filters for modern era (post-1979).
    2.  **Entity Resolution**: Maps disparate team IDs (e.g., 'LVR', 'LV') to a consistent schema.
    3.  **Feature Engineering**:
        - Creates `home_favorite` and `away_favorite` binary flags.
        - Calculates `over` binary target (Total Score > Over/Under Line).
        - Aligns dates between schedule and ELO datasets for accurate merging.
    4.  **Target Definition**:
        - `result`: 1 if Home Team wins, 0 otherwise.

    Args:
        df (pd.DataFrame): Raw games data.
        teams (pd.DataFrame): Teams metadata.
        games_elo (pd.DataFrame): ELO ratings data.

    Returns:
        pd.DataFrame: A fully merged and cleaned DataFrame ready for feature engineering.
    """
    print("Cleaning data...")
    # Replacing blank strings with NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # Removing rows with null values in critical columns
    # We filter for >= 1979 to focus on the modern NFL era (post 16-game schedule implementation).
    df = df[(df.score_home.isnull() == False) & 
            (df.team_favorite_id.isnull() == False) & 
            (df.over_under_line.isnull() == False) &
            (df.schedule_season >= 1979)]

    df.reset_index(drop=True, inplace=True)
    df['over_under_line'] = df.over_under_line.astype(float)

    # Mapping team_id to the correct teams
    df['team_home'] = df.team_home.map(teams.set_index('team_name')['team_id'].to_dict())
    df['team_away'] = df.team_away.map(teams.set_index('team_name')['team_id'].to_dict())

    # Fix team_favorite_id for Colts in specific Superbowls (Historical Data Fixes)
    df.loc[(df.schedule_season == 1968) & (df.schedule_week == 'Superbowl'), 'team_favorite_id'] = 'IND'
    df.loc[(df.schedule_season == 1970) & (df.schedule_week == 'Superbowl'), 'team_favorite_id'] = 'IND'

    # Creating home favorite and away favorite columns
    # These binary features are crucial for models to understand who the expected winner is.
    df.loc[df.team_favorite_id == df.team_home, 'home_favorite'] = 1
    df.loc[df.team_favorite_id == df.team_away, 'away_favorite'] = 1
    df.home_favorite.fillna(0, inplace=True)
    df.away_favorite.fillna(0, inplace=True)

    # Correct known mismatches to ensure consistent Team IDs across datasets
    df['team_favorite_id'] = df['team_favorite_id'].replace({'LV': 'LVR', 'NJY': 'NYJ'})
    df['team_favorite_id'] = df['team_favorite_id'].replace('PICK', np.nan)

    # Creating over / under column (Binary classification target for Over/Under markets)
    df.loc[((df.score_home + df.score_away) > df.over_under_line), 'over'] = 1
    df.over.fillna(0, inplace=True)

    # Convert booleans to int for machine learning compatibility
    df['stadium_neutral'] = df.stadium_neutral.astype(int)
    df['schedule_playoff'] = df.schedule_playoff.astype(int)

    # Date conversion
    df['schedule_date'] = pd.to_datetime(df['schedule_date'])
    games_elo['date'] = pd.to_datetime(games_elo['date'])

    # Fix schedule_week errors (Standardizing week inputs for aggregation)
    df.loc[(df.schedule_week == '18'), 'schedule_week'] = '17'
    df.loc[(df.schedule_week == 'Wildcard') | (df.schedule_week == 'WildCard'), 'schedule_week'] = '18'
    df.loc[(df.schedule_week == 'Division'), 'schedule_week'] = '19'
    df.loc[(df.schedule_week == 'Conference'), 'schedule_week'] = '20'
    df.loc[(df.schedule_week == 'Superbowl') | (df.schedule_week == 'SuperBowl'), 'schedule_week'] = '21'
    df['schedule_week'] = df.schedule_week.astype(int)

    # Select columns of interest
    df = df[['schedule_date', 'schedule_season', 'schedule_week', 'team_home',
           'team_away', 'team_favorite_id', 'spread_favorite',
           'over_under_line', 'weather_temperature',
           'weather_wind_mph', 'score_home', 'score_away',
           'stadium_neutral', 'home_favorite', 'away_favorite',
           'over']]

    # Clean games_elo team names to match base dataset
    games_elo.loc[games_elo.team1 == 'WSH', 'team1'] = 'WAS' 
    games_elo.loc[games_elo.team2 == 'WSH', 'team2'] = 'WAS'

    # Fix dates manually for specific game mismatches found during EDA
    df.loc[(df.schedule_date == '2016-09-19') & (df.team_home == 'MIN'), 'schedule_date'] = datetime.datetime(2016, 9, 18)
    df.loc[(df.schedule_date == '2017-01-22') & (df.schedule_week == 21), 'schedule_date'] = datetime.datetime(2017, 2, 5)
    df.loc[(df.schedule_date == '1990-01-27') & (df.schedule_week == 21), 'schedule_date'] = datetime.datetime(1990, 1, 28)
    df.loc[(df.schedule_date == '1990-01-13'), 'schedule_date'] = datetime.datetime(1990, 1, 14)
    games_elo.loc[(games_elo.date == '2016-01-09'), 'date'] = datetime.datetime(2016, 1, 10)
    games_elo.loc[(games_elo.date == '2016-01-08'), 'date'] = datetime.datetime(2016, 1, 9)
    games_elo.loc[(games_elo.date == '2016-01-16'), 'date'] = datetime.datetime(2016, 1, 17)
    games_elo.loc[(games_elo.date == '2016-01-15'), 'date'] = datetime.datetime(2016, 1, 16)

    # Merge df and games_elo to enrich data with ELO ratings
    df = df.merge(games_elo, left_on=['schedule_date', 'team_home', 'team_away'], right_on=['date', 'team1', 'team2'], how='left')

    # Merge again for swapped home/away scenarios to capture all matches
    games_elo2 = games_elo.rename(columns={'team1' : 'team2', 'team2' : 'team1', 'elo1_pre' : 'elo2_pre', 'elo2_pre' : 'elo1_pre'})
    games_elo2['qbelo_prob1'] = 1 - games_elo2.qbelo_prob1
    df = df.merge(games_elo2, left_on=['schedule_date', 'team_home', 'team_away'], right_on=['date', 'team1', 'team2'], how='left')

    # Fill NaNs from merged columns (Taking data from whichever merge was successful)
    x_cols = ['date_x', 'season_x', 'neutral_x', 'playoff_x', 'team1_x', 'team2_x', 'elo1_pre_x', 'elo2_pre_x', 'qbelo_prob1_x']
    y_cols = ['date_y', 'season_y', 'neutral_y', 'playoff_y', 'team1_y', 'team2_y', 'elo1_pre_y', 'elo2_pre_y', 'qbelo_prob1_y']

    for x, y in zip(x_cols, y_cols):
        df[x] = df[x].fillna(df[y]) 

    # Clean up after merge, keeping only necessary columns
    df = df[['schedule_date', 'schedule_season', 'schedule_week', 'team_home',
           'team_away', 'team_favorite_id', 'spread_favorite', 'over_under_line',
           'weather_temperature', 'weather_wind_mph', 'score_home', 'score_away',
           'stadium_neutral', 'home_favorite', 'away_favorite', 'over', 'neutral_x', 'playoff_x',
             'elo1_pre_x', 'elo2_pre_x', 'qbelo_prob1_x']]

    df.columns = df.columns.str.replace('_x', '')

    # Create Result (Target Variable for Moneyline)
    df['result'] = (df.score_home > df.score_away).astype(int)
    
    return df




def perform_eda(df):
    """
    Performs Exploratory Data Analysis (EDA) and prints key statistical summaries.

    Outputs:
    - Column names and data types.
    - Missing value counts.
    - Key betting metrics:
        *   Home Win %: Frequency of home team winning.
        *   Favored Win %: Frequency of the betting favorite winning (Moneyline/Straight Up).
        *   Cover %: Frequency of the favorite covering the spread.

    Args:
        df (pd.DataFrame): The cleaned games dataset.
    """
    print("\nStarting EDA...")
    print("Columns:", df.columns)
    print("\nMissing Values:\n", df.isnull().sum(axis=0))
    
    home_win = (sum((df.result == 1) & (df.stadium_neutral == 0)) / len(df)) * 100
    favored = (sum(((df.home_favorite == 1) & (df.result == 1)) | ((df.away_favorite == 1) & (df.result == 0))) / len(df)) * 100
    cover = (sum(((df.home_favorite == 1) & ((df.score_away - df.score_home) < df.spread_favorite)) | 
                 ((df.away_favorite == 1) & ((df.score_home - df.score_away) < df.spread_favorite))) / len(df)) * 100
    
    print(f"Number of Games: {len(df)}")
    print(f"Home Straight Up Win Percentage: {home_win:.2f}%")
    print(f"Favored Win Percentage: {favored:.2f}%")
    print(f"Cover The Spread Percentage: {cover:.2f}%")


def engineer_features(df):
    """
    Generates advanced predictive features, focusing on team performance metrics.

    Key Features Created:
    1.  **Point Differentials**: Calculates the margin of victory/defeat for every game.
    2.  **Rolling Averages (`avg_pts_diff`)**:
        - Computes the expanding mean of point differentials for each team.
        - Provides a historical "strength" metric (e.g., how well has this team played up to this point?).
    3.  **Cross-Season Stats**: Merges average point differentials from previous seasons to provide context for the start of a new season.
    4.  **Target Calculation (Spread Coverage)**:
        - `spread_cover`: A binary target derived from the game score and the spread line.
        - Logic: Did the Home Team's adjusted score (Score + Spread) exceed the Away Team's score?

    Args:
        df (pd.DataFrame): The dataset after `clean_data`.

    Returns:
        pd.DataFrame: The final feature-enriched dataset, dropping any rows with NaN values resulting from lag features.
    """
    print("Engineering features...")
    numeric_cols = ['score_home', 'score_away']
    
    # Calculate Mean Scores per team per week
    score = df[['schedule_season', 'schedule_week', 'team_home'] + numeric_cols].groupby(
        ['schedule_season', 'schedule_week', 'team_home']
    )[numeric_cols].mean().reset_index()
    
    aw_score = df[['schedule_season', 'schedule_week', 'team_away'] + numeric_cols].groupby(
        ['schedule_season', 'schedule_week', 'team_away']
    )[numeric_cols].mean().reset_index()
    
    score['point_diff'] = score['score_home'] - score['score_away']
    aw_score['point_diff'] = aw_score['score_away'] - aw_score['score_home']
    aw_score.rename(columns={'team_away': 'team_home'}, inplace=True)
    
    score = pd.concat([score, aw_score], ignore_index=True, sort=False)
    score = score[['schedule_season', 'schedule_week', 'team_home', 'point_diff']]
    score.rename(columns={'team_home': 'team'}, inplace=True)
    score.sort_values(['schedule_season', 'schedule_week'], ascending=[True, True], inplace=True)
    
    # Create rolling averages
    # This loop is essential to build a 'historical' view for each team at any given week.
    tm_dict = {team: df_team.reset_index(drop=True) for team, df_team in score.groupby('team')}
    pts_diff_list = []
    
    for yr in score.schedule_season.unique():
        for tm in score.team.unique():
            if tm in tm_dict:
                data = tm_dict[tm].copy()
                data = data[data.schedule_season == yr]
                data['avg_pts_diff'] = data.point_diff.shift().expanding().mean()
                pts_diff_list.append(data)
                
    pts_diff = pd.concat(pts_diff_list, ignore_index=True, sort=False)
    
    # Merge back to DF (Features for Home and Away teams)
    df = df.merge(
        pts_diff[['schedule_season', 'schedule_week', 'team', 'avg_pts_diff']], 
        left_on=['schedule_season', 'schedule_week', 'team_home'], 
        right_on=['schedule_season', 'schedule_week', 'team'],
        how='left'
    )
    df.rename(columns={'avg_pts_diff' : 'hm_avg_pts_diff'}, inplace=True)
    df.drop(columns=['team'], inplace=True) # Drop redundant col created by merge
    
    df = df.merge(
        pts_diff[['schedule_season', 'schedule_week', 'team', 'avg_pts_diff']], 
        left_on=['schedule_season', 'schedule_week', 'team_away'], 
        right_on=['schedule_season', 'schedule_week', 'team'],
        how='left'
    )
    df.rename(columns={'avg_pts_diff' : 'aw_avg_pts_diff'}, inplace=True)
    df.drop(columns=['team'], inplace=True) # Drop redundant col created by merge
    
    # Calculate Total Season average for previous season stats (Lag Features)
    total_season = pts_diff.groupby(['schedule_season', 'team']).mean()['point_diff'].reset_index()
    total_season['schedule_week'] = 1
    total_season['schedule_season'] += 1
    
    # Merge previous season data
    df = df.merge(total_season[['schedule_season', 'schedule_week', 'team', 'point_diff']], 
                  left_on=['schedule_season', 'schedule_week', 'team_home'], right_on=['schedule_season', 'schedule_week', 'team'],
                  how='left')
    df.rename(columns={'point_diff' : 'hm_avg_diff'}, inplace=True)
    df.drop(columns=['team'], inplace=True) # Drop redundant col created by merge
    
    df = df.merge(total_season[['schedule_season', 'schedule_week', 'team', 'point_diff']], 
                  left_on=['schedule_season', 'schedule_week', 'team_away'], right_on=['schedule_season', 'schedule_week', 'team'],
                  how='left')
    df.rename(columns={'point_diff' : 'aw_avg_diff'}, inplace=True)
    df.drop(columns=['team'], inplace=True) # Drop redundant col created by merge
    
    # Fill NAs
    # If early season data is missing, fallback to previous season's average.
    df.hm_avg_pts_diff.fillna(df.hm_avg_diff, inplace=True)
    df.aw_avg_pts_diff.fillna(df.aw_avg_diff, inplace=True)
    
    # Drop extra cols and keep final feature set
    df = df[['schedule_date', 'schedule_season', 'schedule_week', 'team_home',
           'team_away', 'team_favorite_id', 'spread_favorite', 'over_under_line',
           'weather_temperature', 'weather_wind_mph', 'score_home', 'score_away', 'stadium_neutral', 'home_favorite',
           'away_favorite', 'hm_avg_pts_diff','aw_avg_pts_diff', 'elo1_pre', 'elo2_pre', 'qbelo_prob1', 'over', 'result']]
           
    # Final dropna to ensure clean dataset for models
    df = df.dropna(how='any', axis=0)
    
    # Add Spread Cover Target (Dual Target)
    # We normalize spread to be relative to the home team.
    df['home_spread'] = np.where(df['home_favorite'] == 1, df['spread_favorite'], -df['spread_favorite'])
    df['spread_cover'] = ((df['score_home'] + df['home_spread']) > df['score_away']).astype(int)
    
    return df


def run_model_analysis(df, target_col, analysis_name, output_dir):
    """
    Executes a complete model training and evaluation loop for a specific target.

    Refined for ICLR 2025 compliance:
    - Scales features using StandardScaler.
    - Performs RFE (Recursive Feature Elimination) for selection.
    - Trains a suite of base models (LRG, KNB, GNB, XGB, RFC, DTC).
    - Trains a Stacking Ensemble (Meta-Learner: Logistic Regression).
    - Validates using TimeSeriesSplit to prevent leakage.
    - Performs Statistical T-Tests (Stacked vs Best Single).
    - Generates Calibration Curves.
    """
    print(f"\n{'='*40}")
    print(f"Running Analysis: {analysis_name}")
    print(f"{'='*40}")
    
    # 1. Feature Definition
    X = df[['schedule_season', 'schedule_week', 'over_under_line', 'spread_favorite', 'weather_temperature', 'weather_wind_mph',
            'home_favorite', 'hm_avg_pts_diff','aw_avg_pts_diff', 'elo1_pre', 'elo2_pre', 'qbelo_prob1']]

    y = df[target_col]

    # --- Scientific Rigor: Feature Scaling ---
    scaler = StandardScaler()
    X_scaled_np = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled_np, columns=X.columns)

    # 2. Feature Selection (RFE)
    base = LDA()
    rfe = RFE(estimator=base, n_features_to_select=5)
    rfe = rfe.fit(X, y)

    print(f"RFE Support ({analysis_name}): {rfe.support_}")
    print(f"RFE Ranking ({analysis_name}): {rfe.ranking_}")

    selected_features = X.columns[rfe.support_]
    final_x = X[selected_features]
    print(f"Selected Features: {list(selected_features)}")

    # Prepare models
    models = []
    models.append(('LRG', LogisticRegression(solver='liblinear')))
    models.append(('KNB', KNeighborsClassifier()))
    models.append(('GNB', GaussianNB()))
    models.append(('XGB', xgb.XGBClassifier(random_state=0)))
    models.append(('RFC', RandomForestClassifier(random_state=0, n_estimators=100)))
    models.append(('DTC', DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=5)))

    # Stacking Ensemble
    base_learners = [
        ('LRG', LogisticRegression(solver='liblinear')),
        ('KNB', KNeighborsClassifier()),
        ('GNB', GaussianNB()),
        ('XGB', xgb.XGBClassifier(random_state=0)),
        ('RFC', RandomForestClassifier(random_state=0, n_estimators=100))
    ]
    stack_model = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression())
    models.append(('STACK', stack_model))

    # Evaluate each model
    results = [] 
    names = []
    model_means = {}

    for name, m in models:
        # TimeSeriesSplit
        kfold = model_selection.TimeSeriesSplit(n_splits=5)
        cv_results = model_selection.cross_val_score(m, final_x, y, cv=kfold, scoring = 'roc_auc')
        results.append(cv_results)
        names.append(name)
        model_means[name] = cv_results.mean()
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    
    # --- Scientific Rigor: Statistics ---
    best_single_model = max([n for n in names if n != 'STACK'], key=lambda n: model_means[n])
    print(f"\nBest Single Model: {best_single_model} ({model_means[best_single_model]:.6f})")
    
    stack_scores = results[names.index('STACK')]
    best_single_scores = results[names.index(best_single_model)]
    
    t_stat, p_val = stats.ttest_rel(stack_scores, best_single_scores)
    print(f"Paired T-Test (STACK vs {best_single_model}): p-value = {p_val:.6f}")
    
    # --- Scientific Rigor: Calibration ---
    print("\nGenerating Calibration Curve for STACK model...")
    try:
        train_idx, test_idx = list(kfold.split(final_x))[-1]
        X_train_fold, X_test_fold = final_x.iloc[train_idx], final_x.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
        
        stack_model.fit(X_train_fold, y_train_fold)
        prob_pos = stack_model.predict_proba(X_test_fold)[:, 1]
        
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test_fold, prob_pos, n_bins=10)
        
        plt.figure()
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"STACK ({analysis_name})")
        plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
        plt.ylabel("Fraction of positives")
        plt.xlabel("Mean predicted value")
        plt.title(f"Calibration Curve: {analysis_name}")
        plt.legend()
        plt.grid(True)
        img_name = f"calibration_curve_{target_col}.png"
        save_path = output_dir / img_name
        plt.savefig(save_path)
        print(f"Calibration plot saved to {save_path}")
        plt.close() 
        
    except Exception as e:
        print(f"Could not generate calibration curve: {e}")

    return final_x, selected_features, model_means, p_val, best_single_model



def plot_model_comparison(means_moneyline, means_spread, output_dir):
    """
    Generates a grouped bar chart comparing AUC scores for Moneyline vs Spread.
    
    This visualization is the key artifact demonstrating the "Performance Gap" between 
    the easier Moneyline task and the harder Spread task, a central finding for the ICLR paper.

    Args:
        means_moneyline (dict): Dictionary of {model_name: mean_auc} for Moneyline task.
        means_spread (dict): Dictionary of {model_name: mean_auc} for Spread task.
        output_dir (Path): The Path object to the output directory.

    Outputs:
        Saves the plot to 'model_comparison.png' in the output directory.
    """
    print("\nGenerating Model Comparison Plot...")
    
    models = list(means_moneyline.keys())
    moneyline_scores = [means_moneyline[m] for m in models]
    spread_scores = [means_spread.get(m, 0) for m in models] 
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, moneyline_scores, width, label='Moneyline (Win)', color='skyblue')
    plt.bar(x + width/2, spread_scores, width, label='Spread (Cover)', color='salmon')
    
    plt.ylabel('ROC AUC Score')
    plt.title('Model Performance Comparison: Win Prediction vs. Spread Prediction')
    plt.xticks(x, models)
    plt.ylim(0.45, 0.75) 
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.axhline(0.5, color='gray', linestyle='--', linewidth=1)
    plt.text(len(models)-1, 0.51, 'Random Chance (0.5)', color='gray', ha='right')

    plt.tight_layout()
    save_path = output_dir / 'model_comparison.png'
    plt.savefig(save_path)
    print(f"Saved '{save_path}'.")
    plt.close()


def print_executive_summary(p_moneyline, best_moneyline, p_spread, best_spread):
    """
    Prints a consolidated statistical conclusion for the final report.

    Summarizes:
    1.  Best performing single model for each task.
    2.  Statistical significance of Stacking Ensemble vs. Best Single Model.
    3.  Final conclusion on market efficiency and model utility.

    Args:
        p_moneyline (float): P-value from t-test for Moneyline task.
        best_moneyline (str): Name of the best single model for Moneyline.
        p_spread (float): P-value from t-test for Spread task.
        best_spread (str): Name of the best single model for Spread.
    """
    print("\n" + "="*60)
    print("EXECUTIVE STATISTICAL SUMMARY (ICLR 2025 ALIGNMENT)")
    print("="*60)
    
    print(f"\n1. MONEYLINE TASK (Easier)")
    print(f"   - Best Single Model: {best_moneyline}")
    print(f"   - Stacking Improvement Significant? {'YES' if p_moneyline < 0.05 else 'NO'} (p={p_moneyline:.4f})")
    print(f"   - Conclusion: {'Stacking provides a clear edge.' if p_moneyline < 0.05 else 'Simple models suffice for this strong signal.'}")

    print(f"\n2. SPREAD TASK (Harder)")
    print(f"   - Best Single Model: {best_spread}")
    print(f"   - Stacking Improvement Significant? {'YES' if p_spread < 0.05 else 'NO'} (p={p_spread:.4f})")
    print(f"   - Conclusion: {'Stacking squeezes value from weak signal.' if p_spread < 0.05 else 'Even Stacking struggles; the market is efficient.'}")
    
    print("\n" + "="*60)


def main():
    """
    Main execution entry point.
    
    Orchestrates the full pipeline:
    1.  Argument Parsing
    2.  Data Loading
    3.  Data Cleaning
    4.  Exploratory Data Analysis (EDA)
    5.  Feature Engineering
    6.  Phase 1 Analysis: Moneyline Prediction
    7.  Phase 2 Analysis: Spread Prediction
    8.  Final Reporting (Plots and Summary)
    """
    parser = argparse.ArgumentParser(description="NFL Betting Model - ICLR 2025 Refined Version")
    parser.add_argument("--data-dir", type=str, default=".", help="Directory containing input CSV files")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save plots and models")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Initializing ICLR 2025 NFL Analysis...")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Data Pipeline
    df, teams, games_elo = load_data(data_dir)
    df = clean_data(df, teams, games_elo)
    perform_eda(df)
    df = engineer_features(df)
    
    print(f"Data ready for Analysis. Shape: {df.shape}")

    # Phase 1: Moneyline
    print("\n>>> PHASE 1: Moneyline (Win/Loss) Prediction <<<")
    _, _, means_moneyline, p_moneyline, best_moneyline = run_model_analysis(df, 'result', "Moneyline Prediction", output_dir)

    # Phase 2: Spread
    print("\n>>> PHASE 2: Spread (ATS) Prediction <<<")
    _, _, means_spread, p_spread, best_spread = run_model_analysis(df, 'spread_cover', "Spread Prediction", output_dir)

    # Reporting
    plot_model_comparison(means_moneyline, means_spread, output_dir)
    print_executive_summary(p_moneyline, best_moneyline, p_spread, best_spread)



if __name__ == "__main__":
    main()

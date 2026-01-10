Here’s your content formatted properly as a Markdown file:

---

```markdown
# NFL Betting Model

This project is a machine learning model developed in **`fml.ipynb`** to predict the outcomes of NFL games. The model uses historical game data, betting spreads, and Elo ratings to train a classifier.  
The primary goal is to build a model that outperforms baseline Elo predictions and to evaluate its effectiveness using a simulated betting strategy.

---

## File Structure

- `fml.ipynb` — Main Jupyter Notebook with all data analysis and model training  
- `fml.py` — Standalone Python script version with CLI arguments for flexible execution  
- `NFLMoneyLine_model1.pkl` — Pre-trained model saved after running the notebook  
- `spreadspoke_scores.csv` — Historical NFL scores and betting lines  
- `nfl_teams.csv` — Team information and metadata  
- `nfl_elo.csv` — Historical Elo ratings  
- `README.md` — This file

---

## How to Use This Repository

Follow these steps to set up and run the project on your local machine.

### Prerequisites

- Python 3.7+
- Pip (Python package installer)
- Jupyter Notebook or Jupyter Lab

---

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sanchay-01/Finance_Projects.git
   cd Finance_Projects/nfl_bets

2. **Create a `requirements.txt` file** with the following content:

   ```
   pandas
   numpy
   scikit-learn
   xgboost
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

---

### Running the Code

#### Option 1: Jupyter Notebook
1. **Ensure your data files** (`spreadspoke_scores.csv`, `nfl_teams.csv`, `nfl_elo.csv`) are in the root directory alongside `fml.ipynb`.
2. **Start Jupyter:**

   ```bash
   jupyter lab
   ```

   or

   ```bash
   jupyter notebook
   ```
3. Open `fml.ipynb` and run the cells sequentially from top to bottom.
4. Running the notebook will execute the full analysis, train the model, and save the final, calibrated classifier to `NFLMoneyLine_model1.pkl`.

#### Option 2: Python Script (fml.py)
You can also run the analysis as a standalone Python script with customizable paths:

```bash
python fml.py --data-dir . --output-dir outputs
```

**Available arguments:**
| Argument       | Default   | Description                              |
| :------------- | :-------- | :--------------------------------------- |
| `--data-dir`   | `.`       | Directory containing input CSV files     |
| `--output-dir` | `outputs` | Directory to save plots and models       |
| `--seed`       | `0`       | Random seed for reproducibility          |

---

## Project Overview

The notebook walks through the entire data science pipeline:

1. **Data Loading:**
   Imports historical NFL data from `spreadspoke_scores.csv`, `nfl_teams.csv`, and `nfl_elo.csv`.

2. **Data Cleaning & Preprocessing:**

   * Merges score data with Elo rating data
   * Handles missing values and cleans data types
   * Maps team IDs for consistency
   * Engineers a `result` column (1 for home win, 0 for away win)

3. **Feature Engineering:**

   * Calculates average point differentials for home and away teams (`hm_avg_pts_diff`, `aw_avg_pts_diff`)

4. **Feature Selection:**

   * Uses Recursive Feature Elimination (RFE) with `LinearDiscriminantAnalysis` (LDA)
   * Top 5 features: `spread_favorite`, `home_favorite`, `hm_avg_pts_diff`, `elo2_pre`, `qbelo_prob1`

5. **Model Training:**

   * Training: seasons < 2017
   * Testing: seasons > 2016
   * Combines `XGBClassifier`, `DecisionTreeClassifier`, and `LogisticRegression` in a `VotingClassifier`
   * Wrapped with `CalibratedClassifierCV` (isotonic calibration)

6. **Model Evaluation:**

   * Benchmarked against FiveThirtyEight’s Elo probability (`qbelo_prob1`)
   * Metrics: ROC AUC and Brier Score
   * Simulated betting strategy: bets placed only when model confidence ≥ 60% or ≤ 40%

7. **Model Persistence:**

   * Final trained model saved as `NFLMoneyLine_model1.pkl`

---

## Results

The custom model showed an improvement over baseline Elo predictions on the 2017–2024 test set.

| Metric            | My Model | Elo Results |
| :---------------- | :------: | :---------: |
| **ROC AUC Score** |  0.7188  |    0.7029   |
| **Brier Score**   |  0.2139  |    0.2169   |
| **Betting Win %** |  71.86%  |    69.30%   |

---

## Dependencies

Key Python libraries used in this project:

* `pandas`
* `numpy`
* `scikit-learn`
* `xgboost`
* `pickle`

---

## Future Work

Potential improvements and next steps:

* **Deploy the model:** Create a web app (Streamlit or Flask) to provide predictions for upcoming games.
* **Incorporate more data:** Include player-level stats (injuries, QB performance) and weather data.
* **Hyperparameter Tuning:** Use `GridSearchCV` or `RandomizedSearchCV` to optimize classifiers.
* **Automate Data Ingestion:** Build a script to fetch updated scores and Elo ratings automatically.

---

📘 *Author:[Sanchay Bhutani]*
📅 *Last Updated: November 2025*

```

---

Would you like me to add badges (e.g., Python version, license, notebook status) at the top for a more GitHub-polished look?
```

# Hull Tactical Market Prediction

A machine learning project for predicting S&P 500 tactical positions using market, volatility, sentiment, and macroeconomic features.

## Overview

This repository contains a complete machine learning pipeline for the Hull Tactical Market Prediction competition. The model predicts optimal market positions (0.0 to 2.0x leverage) based on 98 features across market indicators, volatility measures, sentiment indices, and macroeconomic data.

### Key Results

- **Holdout Sharpe Ratio**: 2.89 (on last 180 rows)
- **Total Return**: 18.20% over 121 trading days
- **Volatility Control**: 1.08× market (under 1.2 penalty threshold)
- **Feature Engineering**: Reduced 227 features to 100 high-value features (56% reduction)
- **Model**: LightGBM with heavy regularization for low signal-to-noise data

## Dataset

- **Size**: 9,021 rows × 98 features
- **Time Frame**: ~36 years (approx. 1986-2022)
- **Coverage**: Includes major market events (1987 Crash, Dot-Com Bubble, 2008 GFC, COVID-19)
- **Target**: `market_forward_excess_returns` (next-day S&P 500 excess returns)

### Feature Groups

| Group | Count | Prefix | Description |
|-------|-------|--------|-------------|
| Market | 18 | M | Market indicators |
| Volatility | 13 | V | Volatility measures |
| Sentiment | 12 | S | Sentiment indices |
| Macro | 20 | E | Economic indicators |
| Interest | 9 | I | Interest rates |
| Price | 13 | P | Price-based features |
| Dummy | 9 | D | Binary indicators |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hull-tactical-market-prediction.git
cd hull-tactical-market-prediction

# Install dependencies
pip install -r requirements.txt
```

### Generate Predictions

```bash
# Create submission file
python create_submission.py
```

This will:
1. Load the trained LightGBM model
2. Process test data through the preprocessing pipeline
3. Generate position predictions [0.0, 2.0]
4. Save results to `submission.csv`

## Project Structure

```
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── train.csv                          # Training data (9,021 rows)
├── test.csv                           # Test data
├── submission.csv                     # Generated predictions
│
├── model.py                           # Model inference wrapper
├── preprocessor.py                    # Feature engineering pipeline
├── position_mapper.py                 # Position mapping strategies
├── purged_kfold.py                    # Time-series cross-validation
├── official_metric.py                 # Competition metric calculation
├── create_submission.py               # Submission generator
├── retrain_clean.py                   # Model retraining script
├── phase3_position_mapping.py         # Position strategy optimization
│
├── lgb_model_clean.pkl                # Trained LightGBM model
├── preprocessor_clean.pkl             # Fitted preprocessing pipeline
├── selected_features_clean.pkl        # 100 selected features
├── position_mapper.pkl                # Position mapping strategy
├── feature_importance_clean.csv       # Feature rankings
│
├── analysis_code/                     # Analysis scripts
│   ├── benchmark_analysis.py          # Baseline model comparison
│   ├── calculate_ci.py                # Confidence intervals
│   ├── calculate_rolling.py           # Rolling statistics
│   ├── check_volatility.py            # Volatility analysis
│   ├── create_model_plots.py          # Model visualization
│   ├── create_regime_plot.py          # Regime detection plots
│   ├── fee_analysis.py                # Transaction cost analysis
│   └── validate_holdout.py            # Holdout validation
│
├── documentation/                     # Reports and documentation
│   ├── COMPLETE_ANALYSIS_REPORT.md    # Full analysis report
│   ├── MODEL_DEVELOPMENT_REPORT.md    # Model development details
│   ├── README_STUDY_GUIDE.md          # Study guide
│   ├── presentation_slides.md         # Presentation materials
│   ├── data_analysis.ipynb            # Jupyter notebook
│   └── feature_summary.csv            # Feature statistics
│
├── exploration_plots/                 # Visualization outputs
└── archive/                           # Previous versions and experiments
```

## Methodology

### 1. Feature Engineering

Created 139 new features from 98 base features:

- **Imputation Indicators** (79): `{feature}_missing` flags
- **Lag Features** (24): 1, 2, 5, 10, 20 day lags
- **Rolling Windows** (36): 5, 20, 60-day means and standard deviations

### 2. Feature Selection

Three-step process:
1. **Variance filter** (>0.01): 227 → 218 features
2. **Correlation filter** (<0.95): 218 → 132 features (removed redundant)
3. **Mutual Information**: 132 → **100** features (top predictive power)

### 3. Cross-Validation

**PurgedKFold** with time-based splits:
- 5 folds (first skipped → 4 validation folds)
- 20-day embargo period to prevent autocorrelation leakage
- Proper train/test split (excluded last 180 rows for holdout)

### 4. Model Architecture

**LightGBM** with conservative hyperparameters:

```python
{
    'max_depth': 4,              # Shallow trees
    'num_leaves': 15,             # Conservative
    'learning_rate': 0.02,        # Slow learning
    'feature_fraction': 0.7,      # Feature subsample
    'min_child_samples': 300,     # Prevent overfitting
    'lambda_l1': 1.0,             # L1 regularization
    'lambda_l2': 1.0,             # L2 regularization
}
```

### 5. Position Mapping

**Sign Strategy** (highest Sharpe among 5 tested strategies):
- Position = 0.0 if prediction < 0
- Position = 2.0 if prediction ≥ 0

## Top Features

| Rank | Feature | Importance | Type |
|------|---------|------------|------|
| 1 | M4 | 0.0084 | Market |
| 2 | S2_roll20_mean | 0.0076 | Sentiment (rolling) |
| 3 | M17 | 0.0072 | Market |
| 4 | S5_roll20_mean | 0.0068 | Sentiment (rolling) |
| 5 | M4_lag1 | 0.0057 | Market (lag) |
| 6 | V13 | 0.0055 | Volatility |
| 7 | V13_roll60_std | 0.0050 | Volatility (rolling) |
| 8 | P7 | 0.0049 | Price |
| 9 | I2 | 0.0045 | Interest |
| 10 | M4_lag2 | 0.0043 | Market (lag) |

## Validation Results

### Cross-Validation Performance

| Fold | Sharpe | MSE | Iterations |
|------|--------|-----|------------|
| 1 | -0.15 | 0.000163 | 1 |
| 2 | 0.47 | 0.000136 | 234 |
| 3 | 1.02 | 0.000077 | 237 |
| 4 | 0.08 | 0.000124 | 40 |
| **Mean** | **0.35** | **0.000125** | **128** |

### Holdout Test Results

- **Sharpe Ratio**: 2.89 🎯
- **Total Return**: 18.20%
- **Volatility Ratio**: 1.08 (under 1.2 penalty)
- **Trading Days**: 121 valid samples
- **Position Distribution**: 74% at 0.0, 26% at 2.0

## Competition Metric

**Adjusted Sharpe Formula**:
```
Adjusted Sharpe = Raw Sharpe / (Volatility Penalty × Return Penalty)
```

1. **Volatility Penalty**: `1 + max(0, (Strategy Vol / Market Vol) - 1.2)`
   - Our Result: 1.08 < 1.2 → **No Penalty**

2. **Return Penalty**: `1 + (max(0, Market Return - Strategy Return)² / 100)`
   - Our Result: Strategy > Market → **No Penalty**

**Final Score**: 2.82 (matches raw Sharpe)

## Key Scripts

### Training & Prediction

- **`model.py`**: Model inference wrapper with preprocessing pipeline
- **`preprocessor.py`**: Feature engineering and transformation (imputation, scaling, lag features)
- **`retrain_clean.py`**: Retrain model on clean split (excludes holdout test data)

### Analysis

- **`official_metric.py`**: Calculate competition-specific Sharpe ratio with penalties
- **`phase3_position_mapping.py`**: Test and optimize position mapping strategies
- **`analysis_code/validate_holdout.py`**: Validate model on holdout test set

### Utilities

- **`purged_kfold.py`**: Time-series cross-validation with embargo periods
- **`position_mapper.py`**: Convert predictions to market positions

## Documentation

- **[Complete Analysis Report](documentation/COMPLETE_ANALYSIS_REPORT.md)**: Detailed EDA, feature engineering, and results
- **[Model Development Report](documentation/MODEL_DEVELOPMENT_REPORT.md)**: In-depth model architecture and training
- **[Study Guide](documentation/README_STUDY_GUIDE.md)**: Key concepts and methodology

## Important Notes

### Data Leakage Prevention

 **Proper train/test split implemented**:
- Training: rows [0, 8841] (98% of data)
- Holdout test: last 180 rows (independent validation)
- Embargo periods in cross-validation prevent autocorrelation leakage

### Recommendations for Deployment

1. **Conservative Expectations**: Expect 0.5-1.5 Sharpe on unseen data
2. **Volatility Monitoring**: Use dynamic position sizing if volatility ratio >1.2
3. **Regular Updates**: Retrain monthly to adapt to regime shifts
4. **Risk Management**: Consider capping max position at 1.5 instead of 2.0

## Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
joblib>=1.1.0
```

## Reproducing Results

```bash
# 1. Retrain the model (optional - pre-trained models included)
python retrain_clean.py

# 2. Validate on holdout test set
python analysis_code/validate_holdout.py

# 3. Generate submission
python create_submission.py
```

## License

MIT License

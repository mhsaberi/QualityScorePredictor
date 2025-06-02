# QualityScorePredictor
A machine learning solution for predicting quality scores in a manufacturing dataset using a stacking ensemble of CatBoost and LinearSVR. Developed for a Kaggle competition with a public score of 0.10676 (ranked 10th on the private leaderboard). Includes feature engineering, Optuna tuning, and lessons learned.

## About the Project
This project predicts `quality_score` for manufacturing data using a stacking ensemble. Features include `defect_area`, `plate_length`, and `temperature`. Key aspects:
- Base models: CatBoost and LinearSVR.
- Meta-model: Linear Regression.
- Hyperparameter tuning with Optuna.
- Feature engineering with ratios (e.g., `defect_per_length`) and time-based features (e.g., `day_of_week`).

## Prerequisites
To run this code, you need:
- Python 3.8 or higher
- Required libraries:
  pandas, numpy, scikit-learn, catboost, optuna, joblib

Install them with:
   pip install pandas numpy scikit-learn catboost optuna joblib

- Data files: `train.csv` and `test.csv` (from the Kaggle competition)

## How to Use
1. Place `train.csv` and `test.csv` in the project folder.
2. Open `submission_catboost_linearsvr.ipynb` in Jupyter Notebook or Google Colab.
3. Run the notebook step-by-step to train the model and generate predictions (`catboost_linearsvr_submission.csv`).
4. Use the `predict_new_data` function to predict on new data.

## Results and Lessons
- **Public Score:** 0.10676 (Rank 3 on public leaderboard)
- **Private Score:** Rank 10 (due to poor model choice for non-linear data)
- **Lessons:**
- Non-linear data needs models like SVR with RBF kernel or LightGBM.
- Time-based features (e.g., `day_of_week`) caused overfitting to public data.
- Use higher KFold splits (e.g., 10) for better generalization.

## Next Steps
- Improve the model by adding LightGBM or non-linear SVR.
- Test removing time-based and logarithmic features.
- Explore advanced optimization like genetic algorithms.
  

Big Mart Sales Forecasting System<br>
🚀 Project Overview
This project provides a robust machine learning pipeline for forecasting monthly sales for over 8,500 product–outlet pairs using real-world data from the Big Mart Sales Kaggle competition. Over 80 hours of work went into data cleaning, modeling, and visualization to enable actionable business insights and reliable, production-ready predictions.<br>

📈 Key Achievements<br>
Data scale: Modeled 8,523 sales records from 1,559 products across 10 unique outlets.<br>

Performance: Achieved RMSE as low as 1,051, MAPE down to 57.9%, and R² up to 0.58 on validation, benchmarking within Kaggle’s top ~20% using main features.<br>

Feature Engineering: Encoded 7 categorical columns, 3 numericals, imputed 145 missing item weights, and standardized 5 inconsistent fat categories.

Visualization: Generated 8+ high-impact plots (Actual vs. Predicted Sales, Feature Importance, Residual Analysis, Model Performance Comparison).

Business Impact: Informed inventory, pricing, and outlet targeting decisions through model-driven recommendation.

⚙️ Model Size & Platform Adaptation
Challenge: Model uploads were limited to files ≤5MB, while a full-feature Random Forest (150+ trees, deep splits) reached 194MB.

Solution: Tuned model complexity to create a platform-compliant, small model:

Small model: rf_small.pkl (or rf_small.joblib) – 90 trees, depth 9, just 4.25MB (98% reduction in size), moderate drop in accuracy.

Full-feature model: bigmart_rf_model_full.joblib – the original, unrestricted model (~194MB), for scenarios where upload size is not a concern.

Transparency: Full code and both model artifacts are provided; users may train/generate their own models as needed.

🖥️ How to Use
1. Data Cleaning
python
# See data_cleane.py for cleaning, imputation, and outlier capping.
2. Model Training
python
# Train either the small or full Random Forest models as needed.
# See training scripts for adjustable parameters and notes on file sizes.
3. Visualization
python
# All plot scripts are in the /plots directory or can be reproduced with data_visualizer.py and Plots_For_PPT.py.
4. Model Inference (using the full model by default)
python
import joblib

# Load the full, high-accuracy Random Forest model (≈194MB)
rf_model = joblib.load('bigmart_rf_model_full.joblib')

# Prepare your test/production data (see feature engineering steps in the repo)
# Align your X_test to match the model’s expected columns and encodings

# Predict
predictions = rf_model.predict(X_test)
If platform size constraint exists (e.g., for upload/submission), use the smaller model:

python
rf_model_small = joblib.load('rf_small.pkl')  # or 'rf_small.joblib'
predictions = rf_model_small.predict(X_test)
5. Model Files
rf_small.pkl or rf_small.joblib — Small upload model (≤5MB for certification/platforms)

bigmart_rf_model_full.joblib — Full-feature model, for local/business/analysis use

📦 Other Notes
Both model versions require input features to be preprocessed and encoded identically to training.

For highest reproducibility, use the provided scripts and environment files for consistent dependencies.

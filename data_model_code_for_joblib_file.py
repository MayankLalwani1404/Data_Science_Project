import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import joblib

def train_bigmart_rf_model_joblib(filepath='cleaned_bigmart_data.csv'):
    df = pd.read_csv(filepath)

    # Label encode all categorical features
    encode_cols = [
        'Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier',
        'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'
    ]
    encoders = {}
    for col in encode_cols:
        enc = LabelEncoder()
        df[col + '_enc'] = enc.fit_transform(df[col].astype(str))
        encoders[col] = enc

    feature_cols = (
        ['Item_Weight', 'Item_MRP', 'Outlet_Establishment_Year']
        + [col + '_enc' for col in encode_cols]
    )
    X = df[feature_cols]
    y = df['Item_Outlet_Sales']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Random Forest model
    rf = RandomForestRegressor(n_estimators=450, max_depth=18, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluate performance
    y_pred = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Random Forest Model Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape*100:.2f}%")
    print(f"R2 Score: {r2:.2f}")

    # Save model and encoders using joblib
    joblib.dump(
        {'model': rf, 'encoders': encoders, 'feature_cols': feature_cols},
        'bigmart_rf_model.joblib'
    )
    print("Model saved as 'bigmart_rf_model.joblib'")

    return rf, encoders, feature_cols

# Usage:
rf_model, encoders, feature_cols = train_bigmart_rf_model_joblib()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. Load cleaned training data
df = pd.read_csv('cleaned_bigmart_data.csv')

# 2. Prepare features (make sure the same as used in modeling)
encode_cols = ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 
               'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
for col in encode_cols:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))

feature_cols = ['Item_Weight', 'Item_MRP', 'Outlet_Establishment_Year'] + [col+'_enc' for col in encode_cols]
X = df[feature_cols]
y = df['Item_Outlet_Sales']

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Fit Random Forest model
rf = RandomForestRegressor(n_estimators=450, max_depth=18, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# 5. Calculate residuals (Actual - Predicted)
residuals = y_test - y_pred

# --- PLOT 1: Actual vs Predicted Sales ---
indices = np.argsort(y_test.values)
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(y_test)), y_test.values[indices], label='Actual Sales')
plt.plot(np.arange(len(y_test)), y_pred[indices], label='Predicted Sales')
plt.xlabel('Sample (sorted by true sales)')
plt.ylabel('Item Outlet Sales')
plt.title('Actual vs. Predicted Sales')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted_sales.png')
plt.close()  # Close to save memory

# --- PLOT 2: Feature Importance ---
importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': rf.feature_importances_})
importance_df = importance_df.sort_values(by='Importance', ascending=True)

plt.figure(figsize=(8, 7))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# --- PLOT 3: Forecasted Sales Trend with Confidence Interval ---
window = 20  # Size for rolling window to smooth the line
sorted_preds = np.sort(y_pred)
rolling = pd.Series(sorted_preds).rolling(window, min_periods=1).mean()
ci_lower = rolling * 0.95
ci_upper = rolling * 1.05

plt.figure(figsize=(10, 6))
plt.plot(rolling, label='Forecasted Sales (smoothed)')
plt.fill_between(np.arange(len(rolling)), ci_lower, ci_upper, color='blue', alpha=0.2, label='Confidence Band (~±5%)')
plt.xlabel('Sample (sorted by forecast)')
plt.ylabel('Predicted Sales')
plt.title('Forecasted Sales Trend with Confidence Interval')
plt.legend()
plt.tight_layout()
plt.savefig('forecasted_sales_trend.png')
plt.close()

# --- PLOT 4: Residual Analysis (Errors vs Actual Sales) ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=residuals, alpha=0.4)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residual Analysis: Errors vs Actual Sales')
plt.xlabel('Actual Sales (y_test)')
plt.ylabel('Residual (Actual - Predicted)')
plt.tight_layout()
plt.savefig('residual_analysis_plot.png')
plt.close()

print("All plots saved successfully!")

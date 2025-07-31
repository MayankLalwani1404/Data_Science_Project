import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# 1. Load your cleaned data
df = pd.read_csv('cleaned_bigmart_data.csv')

# 2. Encode categorical variables
encode_cols = ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 
               'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
for col in encode_cols:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))

feature_cols = ['Item_Weight', 'Item_MRP', 'Outlet_Establishment_Year'] + [col+'_enc' for col in encode_cols]
X = df[feature_cols]
y = df['Item_Outlet_Sales']

# 3. Use smaller, lighter Random Forest for file-size constraints
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=90, max_depth=9, random_state=42)
rf.fit(X_train, y_train)

# 4. Save the model with strong compression
with open('rf_small.pkl', 'wb') as f:
    pickle.dump(rf, f, protocol=pickle.HIGHEST_PROTOCOL)

# 5. Check file size
size_mb = os.path.getsize('rf_small.pkl') / (1024 * 1024)
print(f"Saved model size: {size_mb:.2f} MB (should be <5MB)")

# 6. Use this file for upload/certification

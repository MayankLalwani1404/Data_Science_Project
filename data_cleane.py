import pandas as pd
import numpy as np

def clean_bigmart_data(filepath='Train.csv'):
    # Load data
    df = pd.read_csv(filepath)

    # Standardize categorical variable (Item_Fat_Content)
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace(
        {'LF':'Low Fat', 'low fat':'Low Fat', 'reg':'Regular', 
         'Low Fat':'Low Fat', 'Regular':'Regular'}
    )

    # Fill missing Item_Weight with mean weight per Item_Identifier
    df['Item_Weight'] = df.groupby('Item_Identifier')['Item_Weight'].transform(lambda x: x.fillna(x.mean()))
    df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)

    # Fill missing Outlet_Size with mode per Outlet_Type
    outlet_size_mode = df.groupby('Outlet_Type')['Outlet_Size'].agg(lambda x: x.mode().iloc[0])
    missing_outlet_size = df['Outlet_Size'].isnull()
    df.loc[missing_outlet_size, 'Outlet_Size'] = df.loc[missing_outlet_size, 'Outlet_Type'].map(outlet_size_mode)

    # Cap outlier sales at 99th percentile
    cap = df['Item_Outlet_Sales'].quantile(0.99)
    df['Item_Outlet_Sales'] = np.where(df['Item_Outlet_Sales'] > cap, cap, df['Item_Outlet_Sales'])

    # Remove duplicates
    df = df.drop_duplicates()

    # Save cleaned data
    df.to_csv('cleaned_bigmart_data.csv', index=False)
    print("Cleaned data saved as 'cleaned_bigmart_data.csv'")
    return df

# Usage:
df_clean = clean_bigmart_data('Train.csv')

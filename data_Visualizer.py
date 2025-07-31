import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_bigmart_data(df):
    # Sales Distribution
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.histplot(df['Item_Outlet_Sales'], bins=40, color='skyblue')
    plt.title('Sales Distribution (Histogram)')
    plt.xlabel('Item Outlet Sales')
    plt.ylabel('Frequency')

    plt.subplot(1,2,2)
    sns.boxplot(x=df['Item_Outlet_Sales'], color='orange')
    plt.title('Sales Distribution (Boxplot)')
    plt.xlabel('Item Outlet Sales')
    plt.tight_layout()
    plt.show()

    # Revenue by Product Category
    cat_sales = df.groupby('Item_Type')['Item_Outlet_Sales'].sum().sort_values(ascending=False)
    plt.figure(figsize=(12,5))
    sns.barplot(x=cat_sales.index, y=cat_sales.values, palette='viridis')
    plt.title('Total Sales by Product Category')
    plt.xlabel('Item Type')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Revenue by Outlet Location Type
    loc_sales = df.groupby('Outlet_Location_Type')['Item_Outlet_Sales'].sum().sort_values(ascending=False)
    plt.figure(figsize=(8,5))
    sns.barplot(x=loc_sales.index, y=loc_sales.values, palette='magma')
    plt.title('Total Sales by Outlet Location Type')
    plt.xlabel('Outlet Location Type')
    plt.ylabel('Total Sales')
    plt.tight_layout()
    plt.show()

    # Correlation Heatmap
    num_df = df.copy()
    # Encode categoricals for correlation
    for col in ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']:
        num_df[col] = num_df[col].astype('category').cat.codes
    plt.figure(figsize=(8,6))
    sns.heatmap(num_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

    # Outlier Detection (again, boxplot)
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df['Item_Outlet_Sales'], color='lightgreen')
    plt.title('Sales Outlier Boxplot')
    plt.xlabel('Item Outlet Sales')
    plt.tight_layout()
    plt.show()

# Usage:
df_clean = pd.read_csv('cleaned_bigmart_data.csv')
visualize_bigmart_data(df_clean)

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Print the versions of the imported libraries for environment verification
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Seaborn version: {sns.__version__}")


# 1. Load the Dataset

data_path = './data/project table - CAR DETAILS FROM CAR DEKHO.csv'

# Load CSV file
df = pd.read_csv(data_path)

print("\nOriginal DataFrame Head:")
print(df.head()) 
print("\nDataFrame Info:")
df.info() 
print("\nMissing values before preprocessing:")
print(df.isnull().sum()) 



print("\nMissing values after preprocessing")
print(df.isnull().sum())

# Encode categorical
categorical_cols = ['name', 'fuel', 'seller_type', 'transmission', 'owner']

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("\nDataFrame Head after One-Hot Encoding:")
print(df_encoded.head()) 
print(f"Original columns: {df.shape[1]}, Encoded columns: {df_encoded.shape[1]}")



# 2. Data Visualization

plt.figure(figsize=(10, 6)) 
sns.histplot(df_encoded['selling_price'], kde=True) 
plt.title('Distribution of Selling Price') 
plt.xlabel('Selling Price') 
plt.ylabel('Frequency') 
plt.show() 


numerical_cols = ['year', 'km_driven'] 
for col in numerical_cols:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df_encoded[col], y=df_encoded['selling_price']) 
    plt.title(f'{col} vs Selling Price') 
    plt.xlabel(col) 
    plt.ylabel('Selling Price')
    plt.show() 
    
    
    
# Correlation matrix 

non_name_cols = [col for col in df_encoded.columns if not col.startswith('name_')]

if 'selling_price' not in non_name_cols:
    non_name_cols.append('selling_price')


df_for_correlation_heatmap = df_encoded[non_name_cols]

plt.figure(figsize=(12, 10)) 

correlation_matrix_filtered = df_for_correlation_heatmap.corr(numeric_only=True)

sns.heatmap(correlation_matrix_filtered, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Selected Features (Excluding Car Names)') 
plt.show()
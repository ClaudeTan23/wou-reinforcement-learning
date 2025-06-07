import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
file_path = "./dataset/dataset.csv"
df = pd.read_csv(file_path)

# Task 1 Data Loading and Inspection
# Display the first few rows, basic info, and descriptive statistics
head = df.head()

print("\nHead of Dataset (First 5 Rows):")
print(head)

print("\nDataset Info:")
info = df.info()

print("\nDescriptive Statistics:")
describe = df.describe()
print(describe)



# Task 2 Data Preprocessing
# Handle and check missing values
missing_values = df.isnull().sum()
print("\nMissing Values Check:")
print(missing_values, "\n")

for column in df.columns:
    missing_count = df[column].isnull().sum()
    if missing_count > 0:
        print(f"Column '{column}' has {missing_count} missing values.")
    else:
        print(f"Column '{column}' has no missing values.")


# Feature Scaling using MinMaxScaler
scaler = MinMaxScaler()
columns_to_scale = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

print("\nScaled Features (MinMaxScaler applied to area, bedrooms, bathrooms, stories, parking):")
print(df[columns_to_scale].head())


# Label Encoding for categorical variables
label_encoder = LabelEncoder()
categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning']

for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])
print("\nLabel Encoded Features (mainroad, guestroom, basement, hotwaterheating, airconditioning):")
print(df[categorical_columns].head())



# Task 3 Exploratory Data Analysis (EDA)
# Set the plotting style
sns.set_theme(style="whitegrid")
plt.figure(figsize=(16, 12))

# Numerical features to analyze
numerical_features = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']

# Generate Histograms
for i, feature in enumerate(numerical_features):
    plt.subplot(3, 2, i + 1)
    sns.histplot(df[feature], kde=True, bins=30)
    plt.title(f'Distribution of {feature.capitalize()}')

plt.tight_layout()
plt.show()

# Box Plots for Outlier Detection
plt.figure(figsize=(16, 12))
for i, feature in enumerate(numerical_features):
    plt.subplot(3, 2, i + 1)
    sns.boxplot(y=df[feature])
    plt.title(f'Boxplot of {feature.capitalize()}')

plt.tight_layout()
plt.show()



# Task 4 Data Splitting
# Features and target variable
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = label_encoder.fit_transform(df[col])
        print(f"Encoded column: {col}")
        
X = df.drop(columns=['price'])
y = df['price']

# Split the dataset (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

print(f"\nTraining set shape:\n X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"\nTesting set shape:\n X_test: {X_test.shape}, y_test: {y_test.shape}")



# Task 5 Model Architecture
input_dim = X_train.shape[1]

# Build the DNN model
model = Sequential([
    Dense(100, activation='relu', input_shape=(input_dim,)),  # Input + Hidden Layer 1
    Dense(100, activation='relu'),                             # Hidden Layer 2
    Dense(1, activation='linear')                              # Output Layer for regression
])

# Print the model summary
model.summary()



# Task 6 Model Compilation
# Compile the model with specified parameters
model.compile(
    optimizer='rmsprop',              # Optimizer: RMSprop
    loss='mean_squared_error',        # Loss Function: MSE
    metrics=['mean_absolute_error']   # Metric: MAE
)

print("Model compiled has successfully compiled")



# Task 7 Model Training
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.3,  # 30% of training data for validation
    verbose=1
)

plt.figure(figsize=(14, 6))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()

# MAE Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['mean_absolute_error'], label='Training MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Mean Absolute Error Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()



# Task 8 Model Evaluation
# Evaluate the model on the test set
loss, mae = model.evaluate(X_test, y_test, verbose=1)

print("\n--- Model Evaluation on Test Set ---")
print(f"Test Loss (MSE): {loss:.4f}")
print(f"Test MAE: {mae:.4f}")
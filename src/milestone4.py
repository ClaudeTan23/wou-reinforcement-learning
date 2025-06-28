import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras # Explicitly import keras from tensorflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


print(f"Scikit-learn version: {sklearn.__version__}")
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version (TensorFlow's bundled Keras): {tf.__version__}")


data_path = './data/project table - CAR DETAILS FROM CAR DEKHO.csv' 

# Load CSV file
df = pd.read_csv(data_path) 

# Define categorical columns encoded
categorical_cols = ['name', 'fuel', 'seller_type', 'transmission', 'owner']

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


X = df_encoded.drop('selling_price', axis=1) 
y = df_encoded['selling_price'] 

# Convert all features in X to float32
X = X.astype(np.float32)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


# Apply Feature Scaling

numerical_features = ['year', 'km_driven']

scaler = StandardScaler()

X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

print(X_train.head())


# Design and Build the Deep Neural Network (DNN) Architecture

input_dim = X_train.shape[1]

# Create a Sequential model
model = keras.models.Sequential([

    keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),

    keras.layers.Dense(64, activation='relu'),

    keras.layers.Dense(32, activation='relu'),

    keras.layers.Dense(1)
])


model.summary()


# Compile and Train the Model

model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

# Train the model
print("\nTraining DNN model...")
history = model.fit(X_train, y_train,
                    epochs=100, 
                    batch_size=32, 
                    validation_split=0.2,
                    verbose=1) 

# Plotting training history
plt.figure(figsize=(12, 5)) 

plt.subplot(1, 2, 1) 
plt.plot(history.history['loss'], label='Train Loss (MSE)') 
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)') 
plt.title('Model Loss during Training') 
plt.xlabel('Epoch') 
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2) 
plt.plot(history.history['mae'], label='Train MAE') 
plt.plot(history.history['val_mae'], label='Validation MAE') 
plt.title('Model MAE during Training') 
plt.xlabel('Epoch') 
plt.ylabel('MAE') 
plt.legend() 
plt.tight_layout() 
plt.show() 



# Evaluate Model Performance
print("\nEvaluating model performance on test set...")

loss, mae, mse = model.evaluate(X_test, y_test, verbose=0) 
print(f"Test Loss (MSE): {loss:.2f}") 
print(f"Test MAE: {mae:.2f}") 
print(f"Test RMSE: {np.sqrt(mse):.2f}")

# Make predictions
y_pred = model.predict(X_test).flatten()

# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print(f"Test R-squared (R²): {r2:.2f}") 

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6) 

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Prices') 
plt.ylabel('Predicted Prices') 
plt.title('Actual vs Predicted Prices') 
plt.grid(True) 
plt.show()

# Save model
model_path = './models/car_price_dnn.h5'

model.save(model_path) 



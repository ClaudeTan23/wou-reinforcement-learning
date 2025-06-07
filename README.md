# 🏡 Housing Price Prediction using Deep Learning

This project demonstrates the use of a Deep Neural Network (DNN) to predict housing prices based on features such as area, number of bedrooms and bathrooms, presence of amenities, and more. It includes data preprocessing, model training, evaluation, and visualization using Python and Keras.

## 🔧 Tools & Libraries

- **Python 3.10+**
- **Pandas** – for data manipulation
- **NumPy** – for numerical computation
- **Matplotlib & Seaborn** – for data visualization
- **Scikit-learn** – for preprocessing and splitting
- **TensorFlow/Keras** – for building and training the DNN

---

## 📊 Workflow Overview

### Task 1: Data Loading and Inspection
- Load dataset from CSV
- Display structure, data types, and statistics

### Task 2: Data Preprocessing
- Handle missing values
- Normalize numerical features with MinMaxScaler
- Encode categorical features with LabelEncoder

### Task 3: Exploratory Data Analysis (EDA)
- Visualize data distributions and detect outliers using histograms and boxplots

### Task 4: Data Splitting
- Separate features (`X`) and target (`y`)
- Split into training and testing sets (70:30)

### Task 5: Model Architecture
- Build a Sequential model with two hidden layers (100 neurons each)

### Task 6: Model Compilation
- Compile using RMSprop optimizer
- Use Mean Squared Error as loss, and Mean Absolute Error as metric

### Task 7: Model Training
- Train over 100 epochs with 30% validation split
- Visualize loss and MAE across epochs

### Task 8: Model Evaluation
- Evaluate on the test set
- Report final MSE and MAE

---

---
sidebar_position: 1
title: Getting Started
description: Your first steps into AI development with practical examples and hands-on tutorials
---

# Getting Started with AI Workflows

Welcome to your journey into AI development! This tutorial will guide you through setting up your development environment and creating your first AI model. By the end, you'll have a working understanding of basic AI workflows and be ready to tackle more complex projects.

## Prerequisites

Before we begin, ensure you have the following installed:

- **Python 3.8+**: [Download Python](https://www.python.org/downloads/)
- **Git**: [Download Git](https://git-scm.com/downloads)
- **A code editor**: VS Code, PyCharm, or your preferred editor

## Setting Up Your Environment

### 1. Create a Virtual Environment

First, let's create an isolated Python environment:

```bash
# Create a new directory for your project
mkdir ai-first-project
cd ai-first-project

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Required Packages

Install the essential AI and data science libraries:

```bash
# Install core packages
pip install numpy pandas scikit-learn matplotlib seaborn jupyter

# Install deep learning framework (choose one)
pip install tensorflow
# OR
pip install torch torchvision torchaudio

# Install additional utilities
pip install requests tqdm
```

### 3. Verify Installation

Create a test script to verify everything is working:

```python
# test_installation.py
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

print("✅ NumPy version:", np.__version__)
print("✅ Pandas version:", pd.__version__)
print("✅ Scikit-learn version:", sklearn.__version__)

# Test basic operations
data = np.random.randn(100, 2)
df = pd.DataFrame(data, columns=['x', 'y'])
print("✅ Data created successfully!")
print(df.head())
```

Run the test:
```bash
python test_installation.py
```

## Your First AI Model

Let's create a simple machine learning model to predict house prices based on square footage.

### 1. Prepare the Data

Create a dataset with house features:

```python
# house_prices.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate synthetic house data
np.random.seed(42)
n_samples = 1000

# Create features
square_feet = np.random.normal(2000, 500, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)
age = np.random.randint(0, 50, n_samples)

# Create target (house price) with some noise
base_price = 200000
price_per_sqft = 150
price_per_bedroom = 25000
price_per_bathroom = 15000
age_depreciation = 1000

prices = (base_price + 
          square_feet * price_per_sqft + 
          bedrooms * price_per_bedroom + 
          bathrooms * price_per_bathroom - 
          age * age_depreciation + 
          np.random.normal(0, 10000, n_samples))

# Create DataFrame
data = pd.DataFrame({
    'square_feet': square_feet,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'age': age,
    'price': prices
})

print("Dataset created!")
print(data.head())
print(f"\nDataset shape: {data.shape}")
```

### 2. Explore the Data

Add data exploration to understand your dataset:

```python
# Add to house_prices.py
# Data exploration
print("\n=== Data Exploration ===")
print(data.describe())

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())

# Correlation matrix
correlation_matrix = data.corr()
print("\nCorrelation with price:")
print(correlation_matrix['price'].sort_values(ascending=False))

# Visualize relationships
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.scatter(data['square_feet'], data['price'], alpha=0.6)
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.title('Price vs Square Feet')

plt.subplot(2, 2, 2)
plt.scatter(data['bedrooms'], data['price'], alpha=0.6)
plt.xlabel('Bedrooms')
plt.ylabel('Price')
plt.title('Price vs Bedrooms')

plt.subplot(2, 2, 3)
plt.scatter(data['bathrooms'], data['price'], alpha=0.6)
plt.xlabel('Bathrooms')
plt.ylabel('Price')
plt.title('Price vs Bathrooms')

plt.subplot(2, 2, 4)
plt.scatter(data['age'], data['price'], alpha=0.6)
plt.xlabel('Age')
plt.ylabel('Price')
plt.title('Price vs Age')

plt.tight_layout()
plt.savefig('data_exploration.png')
plt.show()
```

### 3. Prepare Features and Target

```python
# Add to house_prices.py
# Prepare features and target
X = data[['square_feet', 'bedrooms', 'bathrooms', 'age']]
y = data['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
```

### 4. Train the Model

```python
# Add to house_prices.py
# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n=== Model Performance ===")
print(f"Mean Squared Error: ${mse:,.2f}")
print(f"R² Score: {r2:.4f}")
print(f"Root Mean Squared Error: ${np.sqrt(mse):,.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.coef_
})
print(f"\nFeature Importance:")
print(feature_importance.sort_values('coefficient', key=abs, ascending=False))
```

### 5. Make Predictions

```python
# Add to house_prices.py
# Make predictions on new data
new_house = pd.DataFrame({
    'square_feet': [2500],
    'bedrooms': [3],
    'bathrooms': [2],
    'age': [10]
})

predicted_price = model.predict(new_house)[0]
print(f"\n=== Prediction ===")
print(f"Predicted price for a 2500 sq ft, 3 bed, 2 bath, 10-year-old house:")
print(f"${predicted_price:,.2f}")
```

## Advanced: Neural Network Model

Let's create a more sophisticated model using a neural network:

```python
# neural_network_model.py
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Prepare data (using the same data from above)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create neural network
model_nn = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model_nn.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Train the model
history = model_nn.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Evaluate
y_pred_nn = model_nn.predict(X_test_scaled).flatten()
mse_nn = mean_squared_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)

print(f"\n=== Neural Network Performance ===")
print(f"Mean Squared Error: ${mse_nn:,.2f}")
print(f"R² Score: {r2_nn:.4f}")

# Compare models
print(f"\n=== Model Comparison ===")
print(f"Linear Regression R²: {r2:.4f}")
print(f"Neural Network R²: {r2_nn:.4f}")
```

## Best Practices

### 1. Data Validation

Always validate your data before training:

```python
def validate_data(data):
    """Validate data quality"""
    issues = []
    
    # Check for missing values
    if data.isnull().any().any():
        issues.append("Missing values detected")
    
    # Check for outliers
    for column in data.select_dtypes(include=[np.number]).columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        outliers = data[(data[column] < Q1 - 1.5 * IQR) | 
                       (data[column] > Q3 + 1.5 * IQR)]
        if len(outliers) > 0:
            issues.append(f"Outliers detected in {column}")
    
    # Check data types
    for column in data.columns:
        if data[column].dtype == 'object':
            issues.append(f"Non-numeric data in {column}")
    
    return issues

# Use the validation
issues = validate_data(data)
if issues:
    print("Data quality issues found:")
    for issue in issues:
        print(f"- {issue}")
else:
    print("✅ Data validation passed!")
```

### 2. Model Persistence

Save your trained model for later use:

```python
import joblib
import pickle

# Save scikit-learn model
joblib.dump(model, 'house_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Save TensorFlow model
model_nn.save('house_price_nn_model')

# Load models later
loaded_model = joblib.load('house_price_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')
loaded_nn_model = tf.keras.models.load_model('house_price_nn_model')
```

### 3. Error Handling

Add proper error handling to your code:

```python
def safe_predict(model, data, scaler=None):
    """Safely make predictions with error handling"""
    try:
        if scaler:
            data_scaled = scaler.transform(data)
            predictions = model.predict(data_scaled)
        else:
            predictions = model.predict(data)
        return predictions
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

# Use the safe prediction function
predictions = safe_predict(model_nn, X_test, scaler)
```

## Next Steps

Congratulations! You've successfully created your first AI model. Here's what to explore next:

### Immediate Next Steps
1. **Experiment with different algorithms**: Try Random Forest, XGBoost, or SVM
2. **Feature engineering**: Create new features from existing ones
3. **Hyperparameter tuning**: Use GridSearchCV or RandomizedSearchCV
4. **Cross-validation**: Implement k-fold cross-validation

### Advanced Topics
- **[Tools Overview](./../tools/overview.md)**: Discover essential AI tools and frameworks
- **[Core Concepts](./../concepts/overview.md)**: Understand AI fundamentals
- **[Contributing Guidelines](./../contributing.md)**: Learn how to contribute

### Resources
- **Datasets**: [Kaggle](https://www.kaggle.com/datasets), [UCI ML Repository](https://archive.ics.uci.edu/ml/)
- **Competitions**: [Kaggle Competitions](https://www.kaggle.com/competitions)
- **Tutorials**: [TensorFlow Tutorials](https://www.tensorflow.org/tutorials), [Scikit-learn Examples](https://scikit-learn.org/stable/auto_examples/)

---

*Remember: AI development is iterative. Start simple, validate your assumptions, and gradually increase complexity as you gain experience.* 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

# Load dataset
df = pd.read_csv('customer_churn.csv')  # Replace with your dataset path

# Display basic info
print(df.head())
print(df.info())

# Example: Encode categorical columns
categorical_cols = ['Gender', 'Geography', 'ContractType']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Define features and target
X = df.drop(['CustomerID', 'Churn'], axis=1)
y = df['Churn']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature importance
import matplotlib.pyplot as plt
xgb.plot_importance(model)
plt.title("Feature Importance")
plt.show()

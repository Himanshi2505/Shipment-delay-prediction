import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load the cleaned dataset
df = pd.read_csv('shipment_data_cleaned.csv')

# Load the saved model columns and scaler
model_columns = joblib.load('model_columns.pkl')
scaler = joblib.load('scaler.pkl')

# Ensure the same columns are present in the dataset
X = df[model_columns]

# Target variable (Delayed)
y = df['Delayed'].map({'Yes': 1, 'No': 0})  # Convert 'Yes'/'No' to 1/0

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data (using the saved scaler)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Train and evaluate Logistic Regression model
log_reg_model = LogisticRegression(max_iter=1000)
log_reg_model.fit(X_train, y_train)

# Predictions
y_pred_log_reg = log_reg_model.predict(X_test)

# Evaluate Logistic Regression performance
print("Logistic Regression Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_log_reg):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_log_reg):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_log_reg):.4f}")

# Train and evaluate Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predictions
y_pred_dt = dt_model.predict(X_test)

# Evaluate Decision Tree performance
print("\nDecision Tree Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_dt):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_dt):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_dt):.4f}")

# Train and evaluate Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate Random Forest performance
print("\nRandom Forest Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_rf):.4f}")

# Save the best performing model (Random Forest, in this case)
joblib.dump(rf_model, 'shipment_delay_model.pkl')

print("\nModel training complete. Best performing model saved as 'shipment_delay_model.pkl'.")

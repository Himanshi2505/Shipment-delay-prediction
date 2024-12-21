import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load cleaned data
df = pd.read_csv('shipment_data_prepared.csv')

# Drop non-numeric columns like 'Shipment ID', 'Origin', 'Destination', etc.
# These columns are not useful for model training
df = df.drop(columns=['Shipment ID', 'Origin', 'Destination', 'Shipment Date', 'Planned Delivery Date', 'Actual Delivery Date'])

# Split data into features and target
X = df.drop(columns=['Delayed'])  # Features
y = df['Delayed'].map({'Yes': 1, 'No': 0})  # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models to experiment with
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# Train and evaluate each model
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Results for {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("-" * 50)
    
    # Save the model
    model_filename = f"{model_name.replace(' ', '_').lower()}_model.pkl"
    joblib.dump(model, model_filename)
    print(f"Model saved as '{model_filename}'.")








# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
# import joblib

# # Load cleaned data
# df = pd.read_csv('shipment_data_prepared.csv')

# # Split data into features and target
# X = df.drop(columns=['Delayed'])
# y = df['Delayed'].map({'Yes': 1, 'No': 0})

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Define models to experiment with
# models = {
#     'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
#     'Decision Tree': DecisionTreeClassifier(random_state=42),
#     'Random Forest': RandomForestClassifier(random_state=42)
# }

# # Train and evaluate each model
# for model_name, model in models.items():
#     print(f"Training {model_name}...")
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
    
#     # Evaluate performance
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
    
#     print(f"Results for {model_name}:")
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"F1 Score: {f1:.4f}")
#     print("Classification Report:\n", classification_report(y_test, y_pred))
#     print("-" * 50)
    
#     # Save the model
#     model_filename = f"{model_name.replace(' ', '_').lower()}_model.pkl"
#     joblib.dump(model, model_filename)
#     print(f"Model saved as '{model_filename}'.")

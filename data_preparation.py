import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load the original dataset
df = pd.read_csv('shipment_data.csv')

# Print columns to check for any missing or incorrect ones
print("Dataset Columns:", df.columns.tolist())

# Drop unnecessary columns
df = df.drop(columns=['Shipment ID', 'Origin', 'Destination', 'Shipment Date', 'Planned Delivery Date', 'Actual Delivery Date'])

# Handle missing values
df = df.dropna()  # Or you can use df.fillna(method='ffill') if you want to impute values

# Handle categorical columns by encoding them
categorical_columns = [col for col in ['Vehicle Type', 'Weather Conditions', 'Traffic Conditions'] if col in df.columns]
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Save the cleaned dataset
df.to_csv('shipment_data_cleaned.csv', index=False)

# Split the dataset into features (X) and target variable (y)
X = df.drop(columns=['Delayed'])  # Features
y = df['Delayed'].map({'Yes': 1, 'No': 0})  # Target

# Save the column names before scaling
columns_before_scaling = X.columns.tolist()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler, model columns, and cleaned data
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(columns_before_scaling, 'model_columns.pkl')

print("Data preparation is complete and saved.")

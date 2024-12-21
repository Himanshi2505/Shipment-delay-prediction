import pandas as pd

# Load the dataset
df = pd.read_csv("shipment_data.csv")

# Handle missing values: Forward fill the missing values
print("Missing values before handling:\n", df.isnull().sum())
df.ffill(inplace=True)
print("\nMissing values after handling:\n", df.isnull().sum())

# Basic info about the dataset
print("\nBasic info about the dataset:")
print(df.info())

# Basic statistics of numeric columns
print("\nBasic statistics:")
print(df.describe())

# Encode categorical variables using the correct column names
df = pd.get_dummies(df, columns=['Vehicle Type', 'Weather Conditions', 'Traffic Conditions'], drop_first=True)

# Calculate and display the correlation matrix (only numeric columns)
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
print("\nCorrelation matrix:")
print(correlation_matrix)

# Save the prepared dataset to a new file
df.to_csv("shipment_data_prepared.csv", index=False)

print("\nData preparation completed. Prepared data saved to shipment_data_prepared.csv.")

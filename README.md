# Shipment Delay Prediction

This repository contains a machine learning project to predict shipment delays based on various features like distance, vehicle type, weather conditions, and traffic conditions. The project includes data preparation, model training, and deployment of a RESTful API for predictions.


## Project Overview

This project focuses on solving the problem of predicting whether a shipment will be delayed or delivered on time. Using machine learning models like Logistic Regression, Decision Tree, and Random Forest, we train and evaluate their performance to select the best-performing model for deployment.



## Setup Instructions

Follow these steps to set up the project:

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/shipment-delay-prediction.git
   cd shipment-delay-prediction
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the data preparation script:
   ```bash
   python data_preparation.py
   ```

5. Train the models:
   ```bash
   python model_training.py
   ```

6. Start the Flask API:
   ```bash
   python app.py
   ```

## Data Preparation

The data preparation process includes:
- Removing unnecessary columns like shipment IDs and dates.
- Handling missing values by dropping rows with nulls.
- Encoding categorical features using one-hot encoding.
- Standardizing numerical features with `StandardScaler`.

The cleaned dataset is saved as `shipment_data_cleaned.csv`.

## Model Training

Three machine learning models were trained and evaluated:
- **Logistic Regression**
  - Accuracy: 0.9111
  - Precision: 1.0000
  - Recall: 0.8798
  - F1 Score: 0.9361

- **Decision Tree**
  - Accuracy: 0.8781
  - Precision: 0.9285
  - Recall: 0.9049
  - F1 Score: 0.9166

- **Random Forest** (Best Model)
  - Accuracy: 0.8748
  - Precision: 0.9233
  - Recall: 0.9060
  - F1 Score: 0.9146

The best-performing model (Random Forest) was saved as `shipment_delay_model.pkl`.

## API Usage

The Flask API provides an endpoint for predicting shipment delays. Use the `/predict` endpoint with a POST request. Example using Postman:

### Request:
```json
POST http://127.0.0.1:5000/predict
{
    "Distance (km)": 1603,
    "Vehicle Type_Lorry": false,
    "Vehicle Type_Trailer": true,
    "Vehicle Type_Truck": false,
    "Weather Conditions_Fog": false,
    "Weather Conditions_Rain": true,
    "Weather Conditions_Storm": false,
    "Traffic Conditions_Light": true,
    "Traffic Conditions_Moderate": false
}
```

### Response:
```json
{
    "prediction": "Delayed"
}
```

## Results

The project demonstrates the following:
1. Random Forest achieved high performance and was selected for deployment.
2. A RESTful API was created for real-time predictions.
3. Predictions are based on features like vehicle type, distance, weather, and traffic conditions.

![prediction](https://github.com/user-attachments/assets/d8180166-79c9-440b-a42f-70f130d2d35b)


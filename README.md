# Shipment Delay Prediction

This repository contains a machine learning project to predict shipment delays based on various features like distance, vehicle type, weather conditions, and traffic conditions. The project includes data preparation, model training, and deployment of a RESTful API for predictions.


## Project Overview

This project focuses on solving the problem of predicting whether a shipment will be delayed or delivered on time. Using machine learning models like Logistic Regression, Decision Tree, and Random Forest, we train and evaluate their performance to select the best-performing model for deployment.



## Setup Instructions

Follow these steps to set up the project:

1. Clone this repository:
   ```bash
   git clone https://github.com/Himanshi2505/Shipment-delay-prediction.git
   cd Shipment-delay-prediction
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
    "prediction": "Yes"
}
```

## Results

The project demonstrates the following:
1. Random Forest achieved high performance and was selected for deployment.
2. A RESTful API was created for real-time predictions.
3. Predictions are based on features like vehicle type, distance, weather, and traffic conditions.

![prediction](https://github.com/user-attachments/assets/b22a9006-88c6-4da3-b520-0f6cd832f904)

## Using Postman to Test the API

Step 1: Start the Flask API Locally
Make sure your Flask application is running. If you're running it locally, open your terminal, navigate to your project folder, and run the following command:

```bash
python app.py
```
By default, Flask runs on http://127.0.0.1:5000/.

Step 2:
Open Postman on your computer. You can download it from Postman's official website if you don't have it installed.

Step 3:
Set Up a POST Request in Postman
URL: In Postman, enter the following URL (assuming you're running Flask locally):
```
http://127.0.0.1:5000/predict

```
HTTP Method: Select POST from the dropdown list of HTTP methods.

Request Body:

Switch to the Body tab in Postman.
Choose raw and then select JSON from the dropdown.
In the text area, enter the data in JSON format that matches the model's expected input. Here's an example:
```json
Copy code
{
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
}
```
Send the Request:

Click on the Send button.
Step 4: View the Response
After sending the request, Postman will display the response in the Response section at the bottom.

If the prediction is successful, the response should look something like this:

```json
Copy code
{
  "prediction": "Yes"
}
```
If there's an error (e.g., missing columns or bad input), the response might contain an error message like this:

```json
Copy code
{
  "error": "Missing required data"
}
```
Additional Tips for Postman
Authorization: If you are deploying the API with authentication, you can set headers for authorization in Postman (e.g., API keys or OAuth tokens).
Save Requests: You can save your request in Postman for reuse by clicking on Save.
Environment Variables: You can create environments in Postman to store different values (like API URLs) for easy switching between local and production environments.



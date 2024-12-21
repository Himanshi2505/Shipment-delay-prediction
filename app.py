from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('logistic_regression_model.pkl')  # Change this to the model you prefer

# Initialize Flask app
app = Flask(__name__)

# Endpoint to predict shipment delay
@app.route('/predict', methods=['POST'])
def predict_delay():
    data = request.get_json(force=True)
    
    # Extract shipment data
    shipment_data = {
        'Origin': data['Origin'],
        'Destination': data['Destination'],
        'Vehicle Type': data['Vehicle Type'],
        'Distance (km)': data['Distance (km)'],
        'Weather Conditions': data['Weather Conditions'],
        'Traffic Conditions': data['Traffic Conditions'],
    }

    # Convert the input data to a DataFrame
    df = pd.DataFrame([shipment_data])

    # Perform any necessary preprocessing here (e.g., encoding categorical variables)
    # Example: df['Weather Conditions'] = df['Weather Conditions'].map({'Clear': 0, 'Rain': 1, 'Fog': 2})
    # Repeat for other categorical columns

    # Prediction
    prediction = model.predict(df)
    result = 'Delayed' if prediction[0] == 1 else 'On Time'

    return jsonify({'Prediction': result})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

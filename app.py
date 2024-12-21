import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model, scaler, and model columns
model = joblib.load('shipment_delay_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json()

        # Create a DataFrame from the incoming JSON data
        input_data = pd.DataFrame(data, index=[0])

        # Ensure that input_data has all the required columns (add missing columns as 0)
        missing_cols = set(model_columns) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0

        # Reorder the columns to match the model's expected order
        input_data = input_data[model_columns]

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make a prediction
        prediction = model.predict(input_data_scaled)

        # Return the result as a JSON response
        result = 'Delayed' if prediction[0] == 1 else 'On Time'
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

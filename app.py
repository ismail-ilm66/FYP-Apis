from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
import pickle
import os
import traceback
import tensorflow as tf
import xgboost as xgb
import logging

from error_handlers import APIException, BadRequest, InternalServerError
from report_generators import generate_crop_report



print(tf.__version__)

# paths to the Crop growth prediction model and label encoder
CROP_GROWTH_PREDICTION_MODEL_PATH = "crop_growth_model.pkl"
CROP_GROWTH_PREDICTION_SCALER_PATH = "scaler.pkl"
CROP_GROWTH_PREDICTION_ENCODER_PATH = "label_encoders.pkl"

# Paths to model and label encoder of crop_prediction model
MODEL_PATH = "crop_cnn_model.h5"
ENCODER_PATH = "label_encoder.pkl"

# Paths to the fertilizer recommendation system  model and label encoder
FERTILIZER_MODEL_PATH = "fertilizer_recommendation_sys.pkl"
FERTILIZER_ENCODER_PATH = "fertilizer_recommendation_sys_le.pkl"

# Check if fertilizer model and encoder files exist
if not os.path.exists(FERTILIZER_MODEL_PATH):
    raise FileNotFoundError(f"Fertilizer model file not found at {FERTILIZER_MODEL_PATH}")

if not os.path.exists(FERTILIZER_ENCODER_PATH):
    raise FileNotFoundError(f"Fertilizer label encoder file not found at {FERTILIZER_ENCODER_PATH}")
# Load fertilizer model and label encoder from .pkl files
try:
    with open(FERTILIZER_MODEL_PATH, 'rb') as f:
        fertilizer_model = pickle.load(f)
    with open(FERTILIZER_ENCODER_PATH, 'rb') as f:
        fertilizer_le = pickle.load(f)
except Exception as e:
    print("Error loading fertilizer model or encoder:", e)
    traceback.print_exc()
    raise

# Feature columns for the fertilizer recommendation model
FERTILIZER_FEATURE_COLUMNS = ['Temperature', 'Humidity', 'Soil Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorus']


# Check if crop_prediction and encoder files exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

if not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(f"Label encoder file not found at {ENCODER_PATH}")

# Load model and label encoder
try:
    model = load_model(MODEL_PATH)
    with open(ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
except Exception as e:
    print("Error loading model or encoder:", e)
    traceback.print_exc()
    raise

#Checking of the Crop growth prediction model and label encoder exists:
# Check if new model files exist
if not os.path.exists(CROP_GROWTH_PREDICTION_MODEL_PATH):
    raise FileNotFoundError(f"New model file not found at {CROP_GROWTH_PREDICTION_MODEL_PATH}")

if not os.path.exists(CROP_GROWTH_PREDICTION_SCALER_PATH):
    raise FileNotFoundError(f"Scaler file not found at {CROP_GROWTH_PREDICTION_SCALER_PATH}")

if not os.path.exists(CROP_GROWTH_PREDICTION_ENCODER_PATH):
    raise FileNotFoundError(f"Label encoder file not found at {CROP_GROWTH_PREDICTION_ENCODER_PATH}")

# Load the new XGBoost model and preprocessing objects
try:
    crop_model = joblib.load(CROP_GROWTH_PREDICTION_MODEL_PATH)
    scaler = joblib.load(CROP_GROWTH_PREDICTION_SCALER_PATH)
    label_encoders = joblib.load(CROP_GROWTH_PREDICTION_ENCODER_PATH)
except Exception as e:
    print("Error loading new model or preprocessing objects:", e)
    traceback.print_exc()
    raise



# Initialize Flask app
app = Flask(__name__)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename='api_errors.log',
    format='%(asctime)s %(levelname)s: %(message)s'
)




@app.errorhandler(APIException)
def handle_api_exception(error):
    logging.info(f"APIException: {error.message}, Code: {error.status_code}")
    return jsonify(error.to_dict()), error.status_code

@app.errorhandler(Exception)
def handle_generic_exception(error):
    logging.error(f"Unexpected error: {str(error)}", exc_info=True)
    return jsonify({
        'error': {
            'message': 'An unexpected error occurred',
            'code': 500
        }
    }), 500

@app.errorhandler(404)
def handle_not_found(error):
    return jsonify({
        'error': {
            'message': 'Resource not found',
            'code': 404
        }
    }), 404

@app.errorhandler(405)
def handle_method_not_allowed(error):
    return jsonify({
        'error': {
            'message': 'Method not allowed',
            'code': 405
        }
    }), 405


#Feature Columns for the crop growth prediction model
FEATURE_COLUMNS = ['Crop', 'Soil_Type', 'Sunlight_Hours', 'Temperature', 'Humidity', 'Water_Frequency', 'Fertilizer_Type']

# Helper function to preprocess input of the crop prediction model
def preprocess_input(data):
    try:
        if not isinstance(data, list) or len(data) != 5:
            raise BadRequest("Please provide exactly 5 features: Temperature, pH, Phosphorus, Nitrogen, Potash")
        
        try:
            features = [float(x) for x in data]
        except (ValueError , TypeError):
            raise BadRequest("Invalid input format. Please provide numeric values.")
        #Validate Ranges
        temp, ph, phosphorus, nitrogen, potash = features
        if not (0 <= temp <= 50):  # Reasonable temperature range (°C)
            raise BadRequest("Temperature must be between 0 and 50°C")
        if not (0 <= ph <= 14):  # pH range
            raise BadRequest("pH must be between 0 and 14")
        if not (0 <= phosphorus <= 1000):  # Arbitrary max for nutrients
            raise BadRequest("Phosphorus must be between 0 and 1000")
        if not (0 <= nitrogen <= 1000):
            raise BadRequest("Nitrogen must be between 0 and 1000")
        if not (0 <= potash <= 1000):
            raise BadRequest("Potash must be between 0 and 1000")


        # Ensure input is in the correct format
        data = np.array(data).reshape(1, 5, 1)  # (batch_size, steps, channels)
        return data
    except Exception as e:
        if isinstance(e , APIException):
            raise
        raise BadRequest(f"Error preprocessing input: {e}")

# Helper function to preprocess input of the crop growth prediction model
def preprocess_crop_input(data):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data], columns=FEATURE_COLUMNS)

        # Encode categorical columns
        categorical_columns = ['Crop', 'Soil_Type', 'Water_Frequency', 'Fertilizer_Type']
        for col in categorical_columns:
            if col in input_df.columns:
                input_df[col] = label_encoders[col].transform(input_df[col])

        # Scale numerical features
        input_scaled = scaler.transform(input_df)

        return input_scaled
    except Exception as e:
        raise ValueError(f"Error preprocessing crop input: {e}")

# Route: Home
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to the Crop Prediction API!",
        "routes": {
            "/predict": "POST - Predict crop based on input features",
            "/predict_crop_growth": "POST - Upload new model or encoder (optional)"
        }
    })

# Route: Predict Crop
@app.route("/predict", methods=["POST"])
def predict_crop():
    try:
        # Get JSON input
        input_data = request.json
        features = input_data.get("features")
        if not features:
            raise BadRequest("Missing 'features' key in JSON")

        
        # Preprocess input
        processed_data = preprocess_input(features)

        # Predict using the model
        prediction = model.predict(processed_data)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])

        try:
            crop_report = generate_crop_report(predicted_label[0])
            return jsonify({
            "input_features": features,
            "predicted_crop": predicted_label[0],
            "idealRangeOfParams": crop_report["idealRangeOfParams"],
            "growingTips": crop_report["growingTips"],
            "growthTimeline": crop_report["growthTimeline"]
        })
        except APIException as e:
            raise
        except Exception as e:
            logging.error(f"Error generating crop report: {str(e)}")
            raise InternalServerError("Failed to generate crop report")

        # Return prediction
       
    except APIException as e:
        raise
    except Exception as e:
        logging.error(f"Predict route error: {str(e)}", exc_info=True)
        raise InternalServerError("Failed to process prediction request")
        

# Route: Predict Crop Growth
@app.route("/predict_crop_growth", methods=["POST"])
def predict_crop_growth():
    try:
        # Get JSON input
        input_data = request.json
        features = input_data.get("features")
        print("Following are the features:")
        print(features)

        if not features or len(features) != len(FEATURE_COLUMNS):
            return jsonify({"error": f"Please provide exactly {len(FEATURE_COLUMNS)} features: {FEATURE_COLUMNS}"}), 400

        # Preprocess input
        processed_data = preprocess_crop_input(features)

        # Predict using the new model
        prediction = crop_model.predict(processed_data)

        # Interpret the result
        result = "Growable" if prediction[0] == 1 else "Not Growable"

        # Return prediction
        return jsonify({
            "input_features": features,
            "prediction": result
        })
    except Exception as e:
        traceback.print_exc()  # Log the error for debugging
        return jsonify({"error": str(e)}), 500


# Route: Predict Fertilizer
@app.route("/predict_fertilizer", methods=["POST"])
def predict_fertilizer():
    try:
        input_data = request.json
        features = input_data.get("features")
        if not features:
            return jsonify({"error": "No features provided"}), 400
        required_features = set(FERTILIZER_FEATURE_COLUMNS)
        provided_features = set(features.keys())
        if not required_features.issubset(provided_features):
            missing = required_features - provided_features
            return jsonify({"error": f"Missing features: {missing}"}), 400
        new_df = pd.DataFrame([features])
        categorical_columns = ['Soil Type', 'Crop Type']
        for col in categorical_columns:
            new_df[col] = new_df[col].astype('category')
        dnew = xgb.DMatrix(new_df, enable_categorical=True)
        pred = fertilizer_model.predict(dnew)
        predicted_label = int(pred[0])
        fertilizer_name = fertilizer_le.inverse_transform([predicted_label])[0]
        return jsonify({
            "input_features": features,
            "predicted_fertilizer": fertilizer_name
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

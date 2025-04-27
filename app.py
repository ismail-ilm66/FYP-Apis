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
from report_generators import generate_crop_report, generate_fertilizer_report, generate_growth_report



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
def a(error):
    logging.info(f"APIException: {error.message}, Code: {error.status_code}")
    return jsonify(error.to_dict()), error.status_code

@app.errorhandler(Exception)
def handle_generic_exception(error):
    logging.error(f"Unexpected error: {str(error)}", exc_info=True)
    return jsonify({
    'status': False,
    'message': 'An unexpected error occurred',
    'code': 500
}), 500

@app.errorhandler(404)
def handle_not_found(error):
    return jsonify({
    'status': False,
    'message': 'Resource not found',
    'code': 404
}), 404

@app.errorhandler(405)
def handle_method_not_allowed(error):
    return jsonify({
    'status': False,
    'message': 'Method not allowed',
    'code': 405
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

# Helper functions to preprocess input of the crop growth prediction model



def validate_growth_input(features):
    # Valid categorical values
    VALID_CROP_TYPES = [
    'Arhar/Tur', 'Bajra', 'Barley', 'Coriander', 'Cotton (Lint)', 'Cowpea (Lobia)', 'Dry Chillies',
    'Garlic', 'Ginger', 'Gram (Chickpea)', 'Groundnut', 'Jowar', 'Linseed (Flax)', 'Maize (Grain)',
    'Maize (Fodder)', 'Masoor (Red Lentil)', 'Moong (Green Gram)', 'Onion', 'Peas & Beans (Pulses)',
    'Potato', 'Ragi (Finger Millet)', 'Rapeseed & Mustard', 'Rice', 'Safflower', 'Sugarcane',
    'Sunflower', 'Turmeric', 'Urad (Black Gram)', 'Urad Bean', 'Wheat'
]
    VALID_SOIL_TYPES = [
    'Silt Loam', 'Loamy', 'Sandy Loam', 'Clayey', 'Well-Drained', 'Slightly Heavy', 'Waterlogged',
    'Sandy', 'Silty Loam', 'Clay Loam', 'Saline-Alkaline', 'Loamy Sand', 'Alluvial', 'Poorly Drained',
    'Silt Clay Loam', 'Red Loam', 'Excessively Alkaline', 'Clay', 'Highly Alkaline', 'Silt Clay'
]
    VALID_WATER_FREQUENCIES = [
    'Moderate', 'Weekly', 'Over-Irrigation', 'Frequent', 'Rare', 'Regular Irrigation', 'Bi-Weekly',
    'Adequate Irrigation', 'Well-Drained', 'Saturated', 'Continuous'
]
    VALID_FERTILIZER_TYPES = [
    'Phosphorus-based', 'Phosphorus-Potassium based', 'Balanced', 'Excessive Fertilizer', 'Imbalanced',
    'Balanced NPK', 'Excessive Nitrogen', 'Low Nitrogen', 'Nitrogen-based', 'Organic', 'High Nitrogen',
    'Nitrogen-Phosphorus based', 'Imbalanced Fertilizer', 'Nitrogen-rich', 'Inadequate',
    'Phosphorus-Sulphur based'
]
    logging.info(f"Validating crop growth input: {features}")
    required_features = set(FEATURE_COLUMNS)
    provided_features = set(features.keys())
    
    if not required_features.issubset(provided_features):
        missing = required_features - provided_features
        raise BadRequest(f"Missing features: {missing}")

    try:
        sunlight_hours = float(features['Sunlight_Hours'])
        temperature = float(features['Temperature'])
        humidity = float(features['Humidity'])

        if not (0 <= sunlight_hours <= 24):
            raise BadRequest("Sunlight Hours must be between 0 and 24 hours")
        if not (0 <= temperature <= 50):
            raise BadRequest("Temperature must be between 0 and 50°C")
        if not (0 <= humidity <= 100):
            raise BadRequest("Humidity must be between 0 and 100%")

        crop = features['Crop']
        soil_type = features['Soil_Type']
        water_frequency = features['Water_Frequency']
        fertilizer_type = features['Fertilizer_Type']

        if crop not in VALID_CROP_TYPES:
            raise BadRequest(f"Invalid Crop. Must be one of: {VALID_CROP_TYPES}")
        if soil_type not in VALID_SOIL_TYPES:
            raise BadRequest(f"Invalid Soil Type. Must be one of: {VALID_SOIL_TYPES}")
        if water_frequency not in VALID_WATER_FREQUENCIES:
            raise BadRequest(f"Invalid Water Frequency. Must be one of: {VALID_WATER_FREQUENCIES}")
        if fertilizer_type not in VALID_FERTILIZER_TYPES:
            raise BadRequest(f"Invalid Fertilizer Type. Must be one of: {VALID_FERTILIZER_TYPES}")

    except (ValueError, TypeError):
        raise BadRequest("Numeric features (Sunlight Hours, Temperature, Humidity) must be numbers")
    except KeyError as e:
        raise BadRequest(f"Missing or invalid feature: {str(e)}")

    return features
def preprocess_crop_input(data):
    try:
        logging.info(f"Preprocessing crop input: {data}")
        # Convert input to DataFrame
        input_df = pd.DataFrame([data], columns=FEATURE_COLUMNS)

        # Validate categorical values against label encoders
        categorical_columns = ['Crop', 'Soil_Type', 'Water_Frequency', 'Fertilizer_Type']
        for col in categorical_columns:
            if col in input_df.columns:
                valid_categories = list(label_encoders[col].classes_)
                if input_df[col].iloc[0] not in valid_categories:
                    raise BadRequest(f"Invalid {col}: {input_df[col].iloc[0]}. Must be one of: {valid_categories}")

        # Encode categorical columns
        for col in categorical_columns:
            input_df[col] = label_encoders[col].transform(input_df[col])

        # Scale numerical features
        input_scaled = scaler.transform(input_df)
        logging.info(f"Preprocessed input: {input_scaled}")

        return input_scaled
    except Exception as e:
        logging.error(f"Preprocessing error: {str(e)}", exc_info=True)
        if isinstance(e, APIException):
            raise
        raise BadRequest(f"Error preprocessing crop input: {str(e)}")



#Helper Function to preprocess the input of the fertilizer prediction model:
def validate_fertilizer_input(features):
    logging.info(f"Validating fertilizer input: {features}")
    required_features = set(FERTILIZER_FEATURE_COLUMNS)
    provided_features = set(features.keys())
    
    # Check for missing features
    if not required_features.issubset(provided_features):
        missing = required_features - provided_features
        raise BadRequest(f"Missing features: {missing}")

    valid_soil_types = [ 'Sandy Loam',
    'Loamy',
    'Sand',
    'Clay Loam',
    'Clay',
    'Sandy',
    'Loamy Sand',
    'Loam',
    'Red Clay Loam',
    'Red Loam',
    'Silty Loam',
    'Alluvial',
    'Black Soil',]  
    valid_crop_types = ['Arhar/Tur',
    'Bajra',
    'Barley',
    'Coriander',
    'Cotton (Lint)',
    'Cowpea (Lobia)',
    'Dry Chillies',
    'Garlic',
    'Ginger',
    'Gram (Chickpea)',
    'Groundnut',
    'Jowar',
    'Linseed (Flax)',
    'Maize (Grain)',
    'Maize (Fodder)',
    'Masoor (Red Lentil)',
    'Moong (Green Gram)',
    'Onion',
    'Peas & Beans (Pulses)',
    'Potato',
    'Ragi (Finger Millet)',
    'Rapeseed & Mustard',
    'Rice',
    'Safflower',
    'Sugarcane',
    'Sunflower',
    'Turmeric',
    'Urad (Black Gram)',
    'Urad Bean',
    'Wheat',] 

    # Validate feature types and ranges
    try:
        # Numeric features
        temp = float(features['Temperature'])
        humidity = float(features['Humidity'])
        soil_moisture = float(features['Soil Moisture'])
        nitrogen = float(features['Nitrogen'])
        potassium = float(features['Potassium'])
        phosphorus = float(features['Phosphorus'])

        # Range checks
        if not (0 <= temp <= 50):
            raise BadRequest("Temperature must be between 0 and 50°C")
        if not (0 <= humidity <= 100):
            raise BadRequest("Humidity must be between 0 and 100%")
        if not (0 <= soil_moisture <= 100):
            raise BadRequest("Soil Moisture must be between 0 and 100%")
        if not (0 <= nitrogen <= 1000):
            raise BadRequest("Nitrogen must be between 0 and 1000 mg/kg")
        if not (0 <= potassium <= 1000):
            raise BadRequest("Potassium must be between 0 and 1000 mg/kg")
        if not (0 <= phosphorus <= 1000):
            raise BadRequest("Phosphorus must be between 0 and 1000 mg/kg")

        # Categorical features
        soil_type = features['Soil Type']
        crop_type = features['Crop Type']
        if soil_type not in valid_soil_types:
            raise BadRequest(f"Invalid Soil Type. Must be one of: {valid_soil_types}")
        if crop_type not in valid_crop_types:
            raise BadRequest(f"Invalid Crop Type. Must be one of: {valid_crop_types}")

    except (ValueError, TypeError):
        raise BadRequest("Numeric features (Temperature, Humidity, Soil Moisture, Nitrogen, Potassium, Phosphorus) must be numbers")
    except KeyError as e:
        raise BadRequest(f"Missing or invalid feature: {str(e)}")

    return features
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
            'status': True,
            "message": "Crop prediction successful",
            'code': 200,
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
    logging.info("Received /predict_crop_growth request")
    
    if not request.is_json:
        raise BadRequest("Content-Type must be application/json")

    input_data = request.get_json()
    if not input_data:
        raise BadRequest("No JSON data provided")

    features = input_data.get("features")
    if not features:
        raise BadRequest("Missing 'features' key in JSON")

    validated_features = validate_growth_input(features)

    processed_data = preprocess_crop_input(validated_features)

    try:
        logging.info("Making crop growth prediction")
        prediction = crop_model.predict(processed_data)
        # Handle scalar or array prediction
        if isinstance(prediction, (list, np.ndarray)):
            predicted_label = int(prediction[0])
        else:
            predicted_label = int(prediction)
        logging.info(f"Predicted label: {predicted_label}")
        result = "Growable" if predicted_label == 1 else "Not Growable"
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}", exc_info=True)
        raise InternalServerError(f"Failed to predict crop growth: {str(e)}")
    
    report = generate_growth_report(validated_features, result)
    return jsonify({
        'status': True,
        'message': 'Crop growth prediction successful',
        'code': 200,
        'input_features': validated_features,
        'prediction': result,
        'recommendations': report,
    })


# Route: Predict Fertilizer
@app.route("/predict_fertilizer", methods=["POST"])
def predict_fertilizer():
    logging.info("Received /predict_fertilizer request")
    
    # Validate content type
    if not request.is_json:
        raise BadRequest("Content-Type must be application/json")

    # Get JSON input
    input_data = request.get_json()
    if not input_data:
        raise BadRequest("No JSON data provided")

    # Get features
    features = input_data.get("features")
    if not features:
        raise BadRequest("Missing 'features' key in JSON")

    # Validate input
    validated_features = validate_fertilizer_input(features)

    # Prepare data for prediction
    new_df = pd.DataFrame([features])
    categorical_columns = ['Soil Type', 'Crop Type']
    for col in categorical_columns:
        new_df[col] = new_df[col].astype('category')
    
    dnew = xgb.DMatrix(new_df, enable_categorical=True)
    
    # Make prediction with try-catch
    try:
        logging.info("Making fertilizer prediction")
        pred = fertilizer_model.predict(dnew)
        predicted_label = int(pred[0])
        fertilizer_name = fertilizer_le.inverse_transform([predicted_label])[0]
    except (xgb.core.XGBoostError, ValueError, IndexError) as e:
        logging.error(f"Prediction or label encoding error: {str(e)}", exc_info=True)
        raise InternalServerError(f"Failed to predict fertilizer: {str(e)}")
    try:
            report = generate_fertilizer_report(fertilizer_name, validated_features['Crop Type'])
            print("Following is the report:")
            print(report)
            return jsonify({
            'status': True,
            'code': 200,
            'message': 'Fertilizer prediction successful',
            'input_features': validated_features,
            'predicted_fertilizer': fertilizer_name,
            'fertilizerDescription': report['fertilizerDescription'],
            'applicationRate': report['applicationRate'],
            'method': report['method'],
            'timing': report['timing'],
            'importantNote': report['importantNote']
        
    })
    except APIException as e:
            raise
    except Exception as e:
            logging.error(f"Error generating Fertilizer report: {str(e)}")
            raise InternalServerError("Failed to generate Fertilizer report")
    
    # Return response


# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

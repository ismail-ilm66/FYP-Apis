from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
import pickle
import os
import traceback
import tensorflow as tf
print(tf.__version__)

# Paths to model and label encoder
MODEL_PATH = "crop_cnn_model.h5"
ENCODER_PATH = "label_encoder.pkl"

# Check if model and encoder files exist
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

# Initialize Flask app
app = Flask(__name__)

# Helper function to preprocess input
def preprocess_input(data):
    try:
        # Ensure input is in the correct format
        data = np.array(data).reshape(1, 5, 1)  # (batch_size, steps, channels)
        return data
    except Exception as e:
        raise ValueError(f"Error preprocessing input: {e}")

# Route: Home
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to the Crop Prediction API!",
        "routes": {
            "/predict": "POST - Predict crop based on input features",
            "/upload": "POST - Upload new model or encoder (optional)"
        }
    })

# Route: Predict Crop
@app.route("/predict", methods=["POST"])
def predict_crop():
    try:
        # Get JSON input
        input_data = request.json
        features = input_data.get("features")

        if not features or len(features) != 5:
            return jsonify({"error": "Please provide exactly 5 features: Temperature, PH, Phosphorous, Nitrogen, Potash"}), 400

        # Preprocess input
        processed_data = preprocess_input(features)

        # Predict using the model
        prediction = model.predict(processed_data)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])

        # Return prediction
        return jsonify({
            "input_features": features,
            "predicted_crop": predicted_label[0]
        })
    except Exception as e:
        traceback.print_exc()  # Log the error for debugging
        return jsonify({"error": str(e)}), 500

# Route: Upload New Model or Encoder (Optional)
@app.route("/upload", methods=["POST"])
def upload_files():
    try:
        model_file = request.files.get("model")
        encoder_file = request.files.get("encoder")

        if model_file:
            model_file.save(MODEL_PATH)
            global model
            model = load_model(MODEL_PATH)

        if encoder_file:
            encoder_file.save(ENCODER_PATH)
            global label_encoder
            with open(ENCODER_PATH, 'rb') as f:
                label_encoder = pickle.load(f)

        return jsonify({"message": "Files uploaded successfully"})
    except Exception as e:
        traceback.print_exc()  # Log the error for debugging
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

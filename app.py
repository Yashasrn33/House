# Import necessary libraries
from fastapi import FastAPI
import torch
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# Define the regression model class
class RegressionModel(torch.nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load model and scalers
MODEL_PATH = "house_price_model.pth"
FEATURE_SCALER_PATH = "feature_scaler.pkl"
TARGET_SCALER_PATH = "target_scaler.pkl"

# Specify the number of features (ensure it matches your dataset's input size)
input_size = 13  # Replace this with the actual number of features in your dataset

# Load the trained model
model = RegressionModel(input_size=input_size)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()  # Set model to evaluation mode

# Load scalers
feature_scaler = joblib.load(FEATURE_SCALER_PATH)
target_scaler = joblib.load(TARGET_SCALER_PATH)

# Initialize FastAPI app
app = FastAPI()

# Root endpoint to provide service details
@app.get("/")
def home():
    return {
        "message": "Welcome to the House Price Prediction Service!",
        "creator": "Yashas Rajanna Naidu",
        "class": "MPS in Applied Machine Intelligence",
        "description": "Predict house prices based on numeric features using a trained neural network model.",
    }

# Define input schema for prediction
class HouseFeatures(BaseModel):
    feature_vector: list  # List of numeric features (must match the number of features used during training)

# Prediction endpoint
@app.post("/predict")
def predict_price(features: HouseFeatures):
    # Convert categorical features to numeric values (e.g., 'yes' -> 1, 'no' -> 0)
    categorical_map = {
        'yes': 1,
        'no': 0,
        'furnished': 1,
        'semi-furnished': 2,
        'unfurnished': 3
    }

    # Transform categorical features
    feature_vector = features.feature_vector
    for i in range(len(feature_vector)):
        if isinstance(feature_vector[i], str) and feature_vector[i] in categorical_map:
            feature_vector[i] = categorical_map[feature_vector[i]]

    # Scale input features using the feature scaler
    scaled_features = feature_scaler.transform([feature_vector])
    tensor_features = torch.tensor(scaled_features, dtype=torch.float32)

    # Predict house price using the trained model
    with torch.no_grad():
        prediction = model(tensor_features).numpy()

    # Rescale prediction back to the original price scale
    predicted_price = target_scaler.inverse_transform(prediction)

    return {"predicted_price": float(predicted_price[0][0])}




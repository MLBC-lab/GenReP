import joblib
import logging
import os
import numpy as np
from .extract_features import smiles_to_features

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GenReP")

class ChemModel:
    def __init__(self, model_path="ensemble_model.joblib", scaffolds_path="scaffolds.npy", scaler_path="scaler.joblib"):
        # Check if model file exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError("Trained model not found. Please specify the correct model_path.")
        
        # Load model
        logger.info("Loading model...")
        self.model = joblib.load(model_path)
        logger.info("Model loaded successfully.")
        
        # Load scaffolds if available
        if os.path.exists(scaffolds_path):
            logger.info("Loading scaffolds...")
            self.scaffolds = np.load(scaffolds_path, allow_pickle=True).tolist()
        else:
            self.scaffolds = None
        
        # Load scaler if available
        if os.path.exists(scaler_path):
            logger.info("Loading scaler...")
            self.scaler = joblib.load(scaler_path)
        else:
            self.scaler = None

    def predict(self, smiles):
        # Convert SMILES to features
        features = smiles_to_features(smiles, self.scaffolds)
        if features is None:
            raise ValueError("Invalid SMILES string provided.")
        
        features = features.reshape(1, -1)  # Reshape for prediction
        
        # Scale if scaler available
        if self.scaler:
            features = self.scaler.transform(features)
        
        # Predict
        logger.info("Making prediction...")
        pred = self.model.predict(features)[0]
        pred_prob = self.model.predict_proba(features)[0][1]  # Probability of class 1
        return pred, pred_prob

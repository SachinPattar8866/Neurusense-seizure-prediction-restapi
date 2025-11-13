# services/model_services.py
import os
import json
import numpy as np
import tensorflow as tf
from config.settings import MODEL_METADATA_PATH, MODEL_HYBRID_PATH, MODEL_CNN_PATH
from utils.errors import APIError

class ModelService:
    """
    Dynamically loads CNN and Hybrid models and provides unified inference methods.
    """

    def __init__(self):
        self.models = {}
        self._load_models()

    def _load_models(self):
        # Ensure metadata exists
        if not os.path.exists(MODEL_METADATA_PATH):
            raise FileNotFoundError(f"Model metadata not found at: {MODEL_METADATA_PATH}")

        with open(MODEL_METADATA_PATH, "r") as f:
            metadata = json.load(f)

        model_entries = metadata.get("models", {})

        if not model_entries:
            raise RuntimeError("No models defined in metadata file.")

        # Load CNN model
        cnn_path = model_entries.get("cnn_best.h5", {}).get("exported_path", MODEL_CNN_PATH)
        if os.path.exists(cnn_path):
            self.models["cnn_best.h5"] = tf.keras.models.load_model(cnn_path)
        else:
            print(f"[WARN] CNN model not found at {cnn_path}")

        # Load Hybrid model
        hybrid_path = model_entries.get("hybrid_best.h5", {}).get("exported_path", MODEL_HYBRID_PATH)
        if os.path.exists(hybrid_path):
            self.models["hybrid_best.h5"] = tf.keras.models.load_model(hybrid_path)
        else:
            print(f"[WARN] Hybrid model not found at {hybrid_path}")

        if not self.models:
            raise RuntimeError("No models were loaded. Check paths and metadata.")

        print(f"[INFO] Loaded models: {list(self.models.keys())}")

    # --- Prediction Methods ---
    def predict(self, image_array: np.ndarray, model_name: str) -> float:
        """Generic inference for specified model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded. Available: {list(self.models.keys())}")
        model = self.models[model_name]
        preds = model.predict(image_array)
        # Handle different output shapes:
        # - multi-class softmax (e.g., shape (1, 3)): return last class (seizure) probability
        # - binary/single-output (shape (1, 1)): return the single value
        try:
            preds = np.array(preds)
            if preds.ndim == 2 and preds.shape[1] > 1:
                # assume seizure is last class
                return float(preds[0, -1])
            elif preds.ndim >= 1:
                return float(preds.flatten()[0])
            else:
                return float(preds)
        except Exception:
            # Fallback to original behaviour
            return float(preds[0][0])

    def predict_with_cnn(self, image_array: np.ndarray) -> float:
        return self.predict(image_array, "cnn_best.h5")

    def predict_with_hybrid(self, image_array: np.ndarray) -> float:
        return self.predict(image_array, "hybrid_best.h5")

    def predict_full(self, image_array: np.ndarray, model_name: str):
        """Return full probability vector (list) from model prediction.

        Returns a list of floats representing the softmax output or a
        single-value list for binary outputs.
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded. Available: {list(self.models.keys())}")
        model = self.models[model_name]
        preds = model.predict(image_array)
        preds = np.array(preds)

        # Normalize shape to 1D list
        if preds.ndim == 2 and preds.shape[0] == 1:
            probs = preds[0].astype(float)
        else:
            probs = preds.flatten().astype(float)

        probs_list = probs.tolist()

        # If the output looks like logits (doesn't sum to ~1) but has multiple classes,
        # apply a softmax so downstream code gets probabilities.
        try:
            if len(probs_list) > 1:
                s = float(sum(probs_list))
                # if sum is near 1, assume already probabilities
                if not (0.999 <= s <= 1.001):
                    # apply softmax (numerically stable)
                    exps = np.exp(probs - np.max(probs))
                    probs = exps / np.sum(exps)
                    probs_list = probs.tolist()
        except Exception:
            # Fall back to raw list if anything goes wrong
            probs_list = [float(x) for x in probs_list]

        return probs_list

    def predict_with_hybrid_full(self, image_array: np.ndarray):
        return self.predict_full(image_array, "hybrid_best.h5")

    def predict_with_cnn_full(self, image_array: np.ndarray):
        return self.predict_full(image_array, "cnn_best.h5")


# Singleton instance
model_service = ModelService()

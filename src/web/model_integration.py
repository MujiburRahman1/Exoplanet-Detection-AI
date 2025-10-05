"""
Model integration for the web application
Loads the trained exoplanet detection model and provides prediction functions
"""

import joblib
import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class ExoplanetModel:
    """Wrapper class for the trained exoplanet detection model"""
    
    def __init__(self, model_path=None):
        """
        Initialize the model
        
        Args:
            model_path: Path to the trained model file
        """
        if model_path is None:
            # Use absolute path for production
            import os
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self.model_path = os.path.join(current_dir, 'trained_exoplanet_model.pkl')
        else:
            self.model_path = model_path
        self.model_data = None
        self.is_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load the trained model from disk"""
        try:
            if os.path.exists(self.model_path):
                self.model_data = joblib.load(self.model_path)
                self.is_loaded = True
                logger.info(f"Model loaded successfully: {self.model_data['model_name']}")
                logger.info(f"Model accuracy: {self.model_data['accuracy']:.4f}")
            else:
                logger.warning(f"Model file not found: {self.model_path}")
                self.is_loaded = False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.is_loaded = False
    
    def predict_single(self, features_dict):
        """
        Predict exoplanet classification from features
        
        Args:
            features_dict: Dictionary with feature values
            
        Returns:
            Dictionary with prediction and confidence
        """
        if not self.is_loaded:
            return {
                'error': 'Model not loaded',
                'prediction': None,
                'confidence': 0.0
            }
        
        try:
            model = self.model_data['model']
            le = self.model_data['label_encoder']
            scaler = self.model_data['scaler']
            features = self.model_data['features']
            
            # Convert to DataFrame
            df = pd.DataFrame([features_dict])
            
            # Ensure all required features are present
            for feature in features:
                if feature not in df.columns:
                    df[feature] = 0.0  # Default value
            
            # Reorder columns
            df = df[features]
            
            # Handle missing values
            df = df.fillna(0.0)
            
            # Scale features if needed
            if self.model_data['model_name'] in ['SVM', 'Neural Network', 'Logistic Regression']:
                df_scaled = scaler.transform(df)
                prediction = model.predict(df_scaled)[0]
                probabilities = model.predict_proba(df_scaled)[0]
            else:
                prediction = model.predict(df)[0]
                probabilities = model.predict_proba(df)[0]
            
            # Decode prediction
            prediction_label = le.inverse_transform([prediction])[0]
            confidence = np.max(probabilities)
            
            # Get class probabilities
            class_probs = {}
            for i, class_name in enumerate(le.classes_):
                class_probs[class_name] = float(probabilities[i])
            
            return {
                'prediction': prediction_label,
                'confidence': float(confidence),
                'class_probabilities': class_probs,
                'model_name': self.model_data['model_name'],
                'model_accuracy': self.model_data['accuracy']
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {
                'error': str(e),
                'prediction': None,
                'confidence': 0.0
            }
    
    def predict_batch(self, df):
        """
        Predict exoplanet classification for a batch of data
        
        Args:
            df: DataFrame with feature columns
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_loaded:
            return pd.DataFrame({'error': ['Model not loaded'] * len(df)})
        
        try:
            model = self.model_data['model']
            le = self.model_data['label_encoder']
            scaler = self.model_data['scaler']
            features = self.model_data['features']
            
            # Prepare features
            X = df.copy()
            
            # Ensure all required features are present
            for feature in features:
                if feature not in X.columns:
                    X[feature] = 0.0  # Default value
            
            # Reorder columns
            X = X[features]
            
            # Handle missing values
            X = X.fillna(0.0)
            
            # Scale features if needed
            if self.model_data['model_name'] in ['SVM', 'Neural Network', 'Logistic Regression']:
                X_scaled = scaler.transform(X)
                predictions = model.predict(X_scaled)
                probabilities = model.predict_proba(X_scaled)
            else:
                predictions = model.predict(X)
                probabilities = model.predict_proba(X)
            
            # Decode predictions
            prediction_labels = le.inverse_transform(predictions)
            confidences = np.max(probabilities, axis=1)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'prediction': prediction_labels,
                'confidence': confidences
            })
            
            # Add class probabilities
            for i, class_name in enumerate(le.classes_):
                results[f'prob_{class_name.lower().replace(" ", "_")}'] = probabilities[:, i]
            
            return results
            
        except Exception as e:
            logger.error(f"Error making batch predictions: {e}")
            return pd.DataFrame({'error': [str(e)] * len(df)})
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if not self.is_loaded:
            return {
                'is_loaded': False,
                'error': 'Model not loaded'
            }
        
        return {
            'is_loaded': True,
            'model_name': self.model_data['model_name'],
            'accuracy': self.model_data['accuracy'],
            'features': self.model_data['features'],
            'n_features': len(self.model_data['features']),
            'classes': list(self.model_data['label_encoder'].classes_),
            'model_path': self.model_path
        }
    
    def get_feature_importance(self):
        """Get feature importance if available"""
        if not self.is_loaded:
            return {}
        
        model = self.model_data['model']
        features = self.model_data['features']
        
        if hasattr(model, 'feature_importances_'):
            importance_dict = {}
            for i, feature in enumerate(features):
                importance_dict[feature] = float(model.feature_importances_[i])
            return importance_dict
        
        return {}

# Global model instance
exoplanet_model = ExoplanetModel()

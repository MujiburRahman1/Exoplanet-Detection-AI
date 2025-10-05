"""
Machine learning models for exoplanet classification.
Implements various algorithms for classifying exoplanets, candidates, and false positives.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import logging
from typing import Dict, List, Tuple, Any, Optional
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExoplanetClassifier:
    """Machine learning classifier for exoplanet detection."""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the classifier.
        
        Args:
            model_type: Type of model to use ('random_forest', 'gradient_boosting', 
                       'svm', 'logistic_regression', 'neural_network')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_trained = False
        
        # Initialize the model based on type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the machine learning model."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42,
                probability=True
            )
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                multi_class='ovr'
            )
        elif self.model_type == 'neural_network':
            self.model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size='auto',
                learning_rate='constant',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the classifier.
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary containing training results and metrics
        """
        logger.info(f"Training {self.model_type} model...")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train_encoded)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_encoded, y_pred)
        
        # Get class names
        class_names = self.label_encoder.classes_
        
        # Generate classification report
        report = classification_report(
            y_test_encoded, y_pred, 
            target_names=class_names, 
            output_dict=True
        )
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train_encoded, 
            cv=5, scoring='accuracy'
        )
        
        self.is_trained = True
        
        results = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y_test_encoded, y_pred).tolist(),
            'class_names': class_names.tolist(),
            'feature_importance': self._get_feature_importance()
        }
        
        logger.info(f"Training completed. Accuracy: {accuracy:.4f}")
        return results
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, prediction_probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Decode labels
        predictions_decoded = self.label_encoder.inverse_transform(predictions)
        
        return predictions_decoded, probabilities
    
    def predict_single(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Make prediction for a single data point.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Dictionary containing prediction and confidence
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert to DataFrame
        X = pd.DataFrame([features])
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in X.columns:
                X[feature] = 0.0  # Default value for missing features
        
        # Reorder columns to match training data
        X = X[self.feature_names]
        
        # Make prediction
        prediction, probabilities = self.predict(X)
        
        # Get confidence (max probability)
        confidence = np.max(probabilities[0])
        
        # Get class probabilities
        class_probs = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            class_probs[class_name] = float(probabilities[0][i])
        
        return {
            'prediction': prediction[0],
            'confidence': float(confidence),
            'class_probabilities': class_probs
        }
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if available."""
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance_dict = {}
                for i, feature in enumerate(self.feature_names):
                    if i < len(self.model.feature_importances_):
                        importance_dict[feature] = float(self.model.feature_importances_[i])
                return importance_dict
            elif hasattr(self.model, 'coef_'):
                # For linear models like SVM, Logistic Regression
                importance_dict = {}
                coef_abs = np.abs(self.model.coef_[0]) if self.model.coef_.ndim > 1 else np.abs(self.model.coef_)
                for i, feature in enumerate(self.feature_names):
                    if i < len(coef_abs):
                        importance_dict[feature] = float(coef_abs[i])
                return importance_dict
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
        return {}
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, 
                            param_grid: Dict[str, List] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X: Feature matrix
            y: Target labels
            param_grid: Parameter grid for tuning
            
        Returns:
            Dictionary containing best parameters and scores
        """
        if param_grid is None:
            param_grid = self._get_default_param_grid()
        
        logger.info(f"Performing hyperparameter tuning for {self.model_type}...")
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_scaled, y_encoded)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        
        return results
    
    def _get_default_param_grid(self) -> Dict[str, List]:
        """Get default parameter grid based on model type."""
        if self.model_type == 'random_forest':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }
        elif self.model_type == 'gradient_boosting':
            return {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            }
        elif self.model_type == 'svm':
            return {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            }
        elif self.model_type == 'logistic_regression':
            return {
                'C': [0.1, 1, 10],
                'max_iter': [100, 500, 1000]
            }
        elif self.model_type == 'neural_network':
            return {
                'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01, 0.1]
            }
        else:
            return {}
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'classes': self.label_encoder.classes_.tolist() if self.is_trained else []
        }

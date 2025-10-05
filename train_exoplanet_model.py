#!/usr/bin/env python3
"""
Complete Exoplanet Detection Model Training Script
Based on your actual NASA Kepler dataset structure
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(csv_file="sample_kepler_data.csv"):
    """
    Load and preprocess the exoplanet dataset
    """
    print(f"📊 Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"✅ Loaded {len(df)} records with {len(df.columns)} features")
    
    # Display target distribution
    print(f"\n🎯 Target Distribution:")
    print(df['koi_pdisposition'].value_counts())
    
    # Select key features for training (excluding IDs and names)
    feature_columns = [
        'koi_period',           # Orbital Period [days]
        'koi_duration',         # Transit Duration [hrs] 
        'koi_depth',            # Transit Depth [ppm]
        'koi_prad',             # Planetary Radius [Earth radii]
        'koi_impact',           # Impact Parameter
        'koi_steff',            # Stellar Effective Temperature [K]
        'koi_srad',             # Stellar Radius [Solar radii]
        'koi_slogg',            # Stellar Surface Gravity
        'koi_model_snr',        # Transit Signal-to-Noise
        'koi_teq',              # Equilibrium Temperature [K]
        'koi_insol',            # Insolation Flux [Earth flux]
        'koi_fpflag_nt',        # Not Transit-Like False Positive Flag
        'koi_fpflag_ss',        # Stellar Eclipse False Positive Flag
        'koi_fpflag_co',        # Centroid Offset False Positive Flag
        'koi_fpflag_ec'         # Ephemeris Match False Positive Flag
    ]
    
    # Check which features are available
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"\n🔧 Using {len(available_features)} features:")
    for feature in available_features:
        print(f"  - {feature}")
    
    # Prepare features and target
    X = df[available_features].copy()
    y = df['koi_pdisposition'].copy()
    
    # Handle missing values
    print(f"\n🧹 Handling missing values...")
    X = X.fillna(X.median())
    
    # Remove any remaining rows with missing values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    print(f"✅ Final dataset: {len(X)} samples, {len(X.columns)} features")
    
    return X, y, available_features

def train_models(X, y, features):
    """
    Train multiple models and compare performance
    """
    print(f"\n🤖 Training Multiple Models...")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models to train
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        
        # Use scaled data for SVM and Neural Network
        if name in ['SVM', 'Neural Network', 'Logistic Regression']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train_scaled if name in ['SVM', 'Neural Network', 'Logistic Regression'] else X_train, 
                                  y_train, cv=5, scoring='accuracy')
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"  ✅ Accuracy: {accuracy:.4f}")
        print(f"  ✅ CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return results, X_test, y_test, le, scaler, features

def save_best_model(results, le, scaler, features):
    """
    Save the best performing model
    """
    print(f"\n💾 Saving Best Model...")
    
    # Find best model based on accuracy
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    
    print(f"🏆 Best Model: {best_model_name}")
    print(f"🎯 Accuracy: {results[best_model_name]['accuracy']:.4f}")
    
    # Save model components
    model_data = {
        'model': best_model,
        'label_encoder': le,
        'scaler': scaler,
        'features': features,
        'model_name': best_model_name,
        'accuracy': results[best_model_name]['accuracy']
    }
    
    joblib.dump(model_data, 'trained_exoplanet_model.pkl')
    print(f"✅ Model saved as 'trained_exoplanet_model.pkl'")
    
    return best_model_name, model_data

def create_prediction_function():
    """
    Create a simple prediction function for testing
    """
    prediction_code = '''
def predict_exoplanet(features_dict):
    """
    Predict exoplanet classification from features
    
    Args:
        features_dict: Dictionary with feature values
        Example: {
            'koi_period': 365.25,
            'koi_duration': 13.0,
            'koi_depth': 14230.9,
            'koi_prad': 13.04,
            'koi_impact': 0.818,
            'koi_steff': 5820.0,
            'koi_srad': 0.964,
            'koi_slogg': 4.457,
            'koi_model_snr': 1339.0,
            'koi_teq': 761.46,
            'koi_insol': 4304.3,
            'koi_fpflag_nt': 0,
            'koi_fpflag_ss': 0,
            'koi_fpflag_co': 0,
            'koi_fpflag_ec': 0
        }
    
    Returns:
        Dictionary with prediction and confidence
    """
    import joblib
    import pandas as pd
    import numpy as np
    
    # Load trained model
    model_data = joblib.load('trained_exoplanet_model.pkl')
    model = model_data['model']
    le = model_data['label_encoder']
    scaler = model_data['scaler']
    features = model_data['features']
    
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
    if model_data['model_name'] in ['SVM', 'Neural Network', 'Logistic Regression']:
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
        'class_probabilities': class_probs
    }

# Example usage:
if __name__ == "__main__":
    # Test prediction
    test_features = {
        'koi_period': 365.25,
        'koi_duration': 13.0,
        'koi_depth': 14230.9,
        'koi_prad': 13.04,
        'koi_impact': 0.818,
        'koi_steff': 5820.0,
        'koi_srad': 0.964,
        'koi_slogg': 4.457,
        'koi_model_snr': 1339.0,
        'koi_teq': 761.46,
        'koi_insol': 4304.3,
        'koi_fpflag_nt': 0,
        'koi_fpflag_ss': 0,
        'koi_fpflag_co': 0,
        'koi_fpflag_ec': 0
    }
    
    result = predict_exoplanet(test_features)
    print("Prediction Result:")
    print(f"Predicted Class: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("Class Probabilities:")
    for class_name, prob in result['class_probabilities'].items():
        print(f"  {class_name}: {prob:.4f}")
'''
    
    with open('predict_exoplanet.py', 'w') as f:
        f.write(prediction_code)
    
    print(f"✅ Prediction function saved as 'predict_exoplanet.py'")

def main():
    """
    Main training pipeline
    """
    print("🚀 Exoplanet Detection Model Training Pipeline")
    print("=" * 50)
    
    # Step 1: Load and preprocess data
    X, y, features = load_and_preprocess_data()
    
    # Step 2: Train models
    results, X_test, y_test, le, scaler, features = train_models(X, y, features)
    
    # Step 3: Save best model
    best_model_name, model_data = save_best_model(results, le, scaler, features)
    
    # Step 4: Create prediction function
    create_prediction_function()
    
    # Step 5: Display results summary
    print(f"\n📊 Training Results Summary:")
    print("-" * 40)
    for name, result in results.items():
        print(f"{name:20s}: {result['accuracy']:.4f} (CV: {result['cv_mean']:.4f} ± {result['cv_std']:.4f})")
    
    print(f"\n🎉 Training Complete!")
    print(f"🏆 Best Model: {best_model_name}")
    print(f"📁 Model saved: trained_exoplanet_model.pkl")
    print(f"🔮 Prediction function: predict_exoplanet.py")
    
    print(f"\n📋 Next Steps:")
    print(f"1. Test the model: python predict_exoplanet.py")
    print(f"2. Integrate with your web app")
    print(f"3. Upload new data for predictions")

if __name__ == "__main__":
    main()

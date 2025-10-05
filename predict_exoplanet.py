
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

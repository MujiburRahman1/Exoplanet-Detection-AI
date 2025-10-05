"""
Flask web application for exoplanet detection.
Provides a user interface for uploading data and viewing predictions.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import json
from werkzeug.utils import secure_filename
import logging
from datetime import datetime

# Import our custom modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.data_loader import ExoplanetDataLoader
from src.models.exoplanet_classifier import ExoplanetClassifier
from src.web.model_integration import exoplanet_model
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
           template_folder='../../templates',
           static_folder='../../static')
app.config['SECRET_KEY'] = 'exoplanet-detection-secret-key'
app.config['UPLOAD_FOLDER'] = '../../uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Enable CORS
CORS(app)

# Global variables
# Use absolute path for production deployment
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
data_loader = ExoplanetDataLoader(data_dir=data_dir)
classifier = None
model_info = {}

def analyze_uploaded_data(df, classifier):
    """Analyze uploaded data and make predictions."""
    try:
        # Get feature columns that match our training data
        feature_columns = [
            'koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_impact',
            'koi_steff', 'koi_srad', 'koi_slogg', 'koi_model_snr', 'koi_teq',
            'koi_insol', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec'
        ]
        
        # Map common column names to our expected features
        column_mapping = {
            'Orbital Period (days)': 'koi_period',
            'Transit Duration (hours)': 'koi_duration', 
            'Transit Depth (ppm)': 'koi_depth',
            'Planetary Radius (Earth radii)': 'koi_prad',
            'Impact Parameter': 'koi_impact',
            'Stellar Temperature (K)': 'koi_steff',
            'Stellar Radius (Solar radii)': 'koi_srad',
            'Signal to Noise Ratio': 'koi_model_snr'
        }
        
        # Create a working dataframe
        work_df = df.copy()
        
        # Map columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in work_df.columns:
                work_df[new_name] = work_df[old_name]
        
        # Select available features
        available_features = [col for col in feature_columns if col in work_df.columns]
        
        if len(available_features) < 5:
            raise ValueError(f"Not enough features found. Available: {available_features}")
        
        # Prepare data for prediction
        X = work_df[available_features].copy()
        
        # Handle missing values
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 0)
        
        # Make predictions
        predictions = []
        for idx, row in X.iterrows():
            try:
                # Convert to DataFrame for single prediction
                single_row = pd.DataFrame([row])
                pred, proba = classifier.predict(single_row)
                
                # Get confidence
                confidence = float(np.max(proba[0]))
                
                predictions.append({
                    'row_index': idx,
                    'prediction': pred[0],
                    'confidence': confidence,
                    'features_used': available_features
                })
            except Exception as e:
                predictions.append({
                    'row_index': idx,
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'error': str(e)
                })
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error analyzing uploaded data: {e}")
        raise e

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html', model_info=model_info)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload for data analysis."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Load and analyze the uploaded data
                if filename.endswith('.csv'):
                    df = pd.read_csv(filepath)
                else:
                    df = pd.read_excel(filepath)
                
                # Analyze the uploaded data if model is available
                analysis_results = None
                if classifier and classifier.is_trained:
                    try:
                        # Try to make predictions on uploaded data
                        predictions = analyze_uploaded_data(df, classifier)
                        analysis_results = {
                            'total_rows': len(df),
                            'predictions': predictions,
                            'candidates_found': len([p for p in predictions if p.get('prediction') == 'CANDIDATE']),
                            'false_positives': len([p for p in predictions if p.get('prediction') == 'FALSE POSITIVE']),
                            'analysis_available': True
                        }
                    except Exception as e:
                        logger.warning(f"Could not analyze uploaded data: {e}")
                        analysis_results = {
                            'analysis_available': False,
                            'error': str(e)
                        }
                else:
                    analysis_results = {
                        'analysis_available': False,
                        'error': 'No trained model available for analysis'
                    }
                
                # Store the uploaded data for analysis
                session_data = {
                    'filename': filename,
                    'filepath': filepath,
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'preview': df.head().to_html(classes='table table-striped'),
                    'analysis': analysis_results
                }
                
                return render_template('upload_success.html', data=session_data)
                
            except Exception as e:
                flash(f'Error processing file: {str(e)}')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload CSV or Excel files.')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/analyze', methods=['POST'])
def analyze_data():
    """Analyze uploaded data and make predictions."""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # Load the data
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        # Check if classifier is trained
        if not classifier or not classifier.is_trained:
            return jsonify({'error': 'Model not trained. Please train a model first.'}), 400
        
        # Preprocess the data
        processed_df, _ = data_loader.preprocess_data(df)
        
        # Make predictions
        predictions, probabilities = classifier.predict(processed_df)
        
        # Create results
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            confidence = np.max(prob)
            class_probs = {}
            for j, class_name in enumerate(classifier.label_encoder.classes_):
                class_probs[class_name] = float(prob[j])
            
            results.append({
                'index': i,
                'prediction': pred,
                'confidence': float(confidence),
                'class_probabilities': class_probs
            })
        
        return jsonify({
            'success': True,
            'results': results,
            'summary': {
                'total_predictions': len(results),
                'predictions_by_class': pd.Series(predictions).value_counts().to_dict()
            }
        })
        
    except Exception as e:
        logger.error(f"Error analyzing data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict')
def predict():
    """Prediction interface page."""
    return render_template('predict.html')

@app.route('/train', methods=['GET', 'POST'])
def train_model():
    """Train the machine learning model."""
    global classifier, model_info
    
    if request.method == 'POST':
        try:
            data = request.get_json()
            model_type = data.get('model_type', 'random_forest')
            dataset = data.get('dataset', 'kepler')
            
            # Initialize classifier
            classifier = ExoplanetClassifier(model_type=model_type)
            
            # Load dataset
            if dataset == 'kepler':
                df = data_loader.load_kepler_data()
                target_column = 'koi_pdisposition'  # Disposition Using Kepler Data
            elif dataset == 'tess':
                df = data_loader.load_tess_data()
                target_column = 'tfopwg_disposition'  # TESS Follow-up Observing Program Working Group Disposition
            elif dataset == 'k2':
                df = data_loader.load_k2_data()
                target_column = 'disposition'  # Archive Disposition
            else:
                return jsonify({'error': 'Invalid dataset selected'}), 400
            
            if df.empty:
                return jsonify({'error': f'Dataset {dataset} not found or empty'}), 400
            
            # Preprocess data
            X, y = data_loader.preprocess_data(df, target_column)
            
            if y is None:
                return jsonify({'error': f'Target column {target_column} not found'}), 400
            
            # Train the model
            results = classifier.train(X, y)
            
            # Update model info
            model_info = classifier.get_model_info()
            model_info.update({
                'training_results': results,
                'dataset_used': dataset,
                'trained_at': datetime.now().isoformat()
            })
            
            # Format training results for frontend
            formatted_results = {
                'accuracy': results.get('accuracy', 0),
                'cv_mean': results.get('cv_mean', 0),
                'cv_std': results.get('cv_std', 0),
                'classification_report': results.get('classification_report', {}),
                'confusion_matrix': results.get('confusion_matrix', []),
                'class_names': results.get('class_names', []),
                'feature_importance': results.get('feature_importance', {})
            }
            
            return jsonify({
                'success': True,
                'model_info': model_info,
                'training_results': formatted_results
            })
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return jsonify({'error': str(e)}), 500
    
    return render_template('train.html')

@app.route('/model_info')
def get_model_info():
    """Get current model information."""
    # Get info from trained model
    trained_model_info = exoplanet_model.get_model_info()
    
    # Merge with existing model_info
    combined_info = {**model_info, **trained_model_info}
    
    return jsonify(combined_info)

@app.route('/datasets')
def get_datasets():
    """Get information about available datasets."""
    try:
        datasets = data_loader.load_all_datasets()
        info = data_loader.get_dataset_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_single', methods=['POST'])
def predict_single():
    """Make prediction for a single data point."""
    try:
        data = request.get_json()
        features = data.get('features', {})
        
        # Convert string values to float where possible
        processed_features = {}
        for key, value in features.items():
            try:
                processed_features[key] = float(value)
            except (ValueError, TypeError):
                processed_features[key] = 0.0
        
        # Make prediction using trained model
        result = exoplanet_model.predict_single(processed_features)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 400
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Error making single prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/hyperparameter_tuning', methods=['POST'])
def hyperparameter_tuning():
    """Perform hyperparameter tuning."""
    try:
        if not classifier:
            return jsonify({'error': 'No classifier initialized'}), 400
        
        data = request.get_json()
        dataset = data.get('dataset', 'kepler')
        
        # Load dataset
        if dataset == 'kepler':
            df = data_loader.load_kepler_data()
            target_column = 'koi_pdisposition'  # Disposition Using Kepler Data
        elif dataset == 'tess':
            df = data_loader.load_tess_data()
            target_column = 'tfopwg_disposition'  # TESS Follow-up Observing Program Working Group Disposition
        elif dataset == 'k2':
            df = data_loader.load_k2_data()
            target_column = 'disposition'  # Archive Disposition
        else:
            return jsonify({'error': 'Invalid dataset selected'}), 400
        
        if df.empty:
            return jsonify({'error': f'Dataset {dataset} not found or empty'}), 400
        
        # Preprocess data
        X, y = data_loader.preprocess_data(df, target_column)
        
        if y is None:
            return jsonify({'error': f'Target column {target_column} not found'}), 400
        
        # Perform hyperparameter tuning
        results = classifier.hyperparameter_tuning(X, y)
        
        return jsonify({
            'success': True,
            'tuning_results': results
        })
        
    except Exception as e:
        logger.error(f"Error in hyperparameter tuning: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/save_model', methods=['POST'])
def save_model():
    """Save the trained model."""
    try:
        if not classifier or not classifier.is_trained:
            return jsonify({'error': 'No trained model to save'}), 400
        
        data = request.get_json()
        filename = data.get('filename', f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib')
        
        filepath = os.path.join('../../models', filename)
        os.makedirs('../../models', exist_ok=True)
        
        classifier.save_model(filepath)
        
        return jsonify({
            'success': True,
            'message': f'Model saved as {filename}',
            'filepath': filepath
        })
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/load_model', methods=['POST'])
def load_model():
    """Load a saved model."""
    global classifier, model_info
    
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        filepath = os.path.join('../../models', filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Model file not found'}), 404
        
        # Load the model
        classifier = ExoplanetClassifier()
        classifier.load_model(filepath)
        
        # Update model info
        model_info = classifier.get_model_info()
        model_info['loaded_at'] = datetime.now().isoformat()
        model_info['filename'] = filename
        
        return jsonify({
            'success': True,
            'message': f'Model loaded from {filename}',
            'model_info': model_info
        })
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

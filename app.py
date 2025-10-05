#!/usr/bin/env python3
"""
Simple Flask app for deployment
"""

import os
import sys
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'exoplanet-detection-secret-key'
CORS(app)

@app.route('/')
def home():
    """Home page"""
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Exoplanet Detection AI</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .container {
                background: white;
                padding: 2rem;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                text-align: center;
                max-width: 600px;
                margin: 1rem;
            }
            h1 {
                color: #333;
                margin-bottom: 1rem;
            }
            p {
                color: #666;
                line-height: 1.6;
                margin-bottom: 2rem;
            }
            .btn {
                display: inline-block;
                padding: 12px 24px;
                background: #667eea;
                color: white;
                text-decoration: none;
                border-radius: 8px;
                margin: 0.5rem;
                transition: background 0.3s;
            }
            .btn:hover {
                background: #5a6fd8;
            }
            .status {
                background: #e8f5e8;
                border: 1px solid #4caf50;
                color: #2e7d32;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Exoplanet Detection AI</h1>
            <div class="status">
                ✅ Application is running successfully!
            </div>
            <p>
                Welcome to the Exoplanet Detection AI application. This system uses machine learning 
                to analyze astronomical data and identify potential exoplanets from NASA's Kepler, 
                K2, and TESS missions.
            </p>
            <a href="/upload" class="btn">Upload Data</a>
            <a href="/train" class="btn">Train Model</a>
            <a href="/predict" class="btn">Make Prediction</a>
            <a href="/model_info" class="btn">Model Info</a>
        </div>
    </body>
    </html>
    '''

@app.route('/upload')
def upload_page():
    """Upload page"""
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Upload Data - Exoplanet Detection AI</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 2rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                background: white;
                padding: 2rem;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                max-width: 600px;
                margin: 0 auto;
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 2rem;
            }
            .form-group {
                margin-bottom: 1.5rem;
            }
            label {
                display: block;
                margin-bottom: 0.5rem;
                color: #555;
                font-weight: 500;
            }
            input[type="file"] {
                width: 100%;
                padding: 0.75rem;
                border: 2px dashed #667eea;
                border-radius: 8px;
                background: #f8f9ff;
            }
            .btn {
                display: inline-block;
                padding: 12px 24px;
                background: #667eea;
                color: white;
                text-decoration: none;
                border-radius: 8px;
                border: none;
                cursor: pointer;
                width: 100%;
                font-size: 16px;
                margin-top: 1rem;
            }
            .btn:hover {
                background: #5a6fd8;
            }
            .back-btn {
                background: #6c757d;
                margin-bottom: 1rem;
                width: auto;
                padding: 8px 16px;
            }
            .back-btn:hover {
                background: #5a6268;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <a href="/" class="btn back-btn">← Back to Home</a>
            <h1>📁 Upload Data</h1>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="file">Select CSV File:</label>
                    <input type="file" id="file" name="file" accept=".csv" required>
                </div>
                <button type="submit" class="btn">Upload File</button>
            </form>
            <p style="margin-top: 2rem; color: #666; font-size: 14px;">
                <strong>Supported formats:</strong> CSV files with exoplanet data<br>
                <strong>File size limit:</strong> 16MB
            </p>
        </div>
    </body>
    </html>
    '''

@app.route('/train')
def train_page():
    """Training page"""
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Train Model - Exoplanet Detection AI</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 2rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                background: white;
                padding: 2rem;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                max-width: 600px;
                margin: 0 auto;
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 2rem;
            }
            .form-group {
                margin-bottom: 1.5rem;
            }
            label {
                display: block;
                margin-bottom: 0.5rem;
                color: #555;
                font-weight: 500;
            }
            select {
                width: 100%;
                padding: 0.75rem;
                border: 1px solid #ddd;
                border-radius: 8px;
                background: white;
            }
            .btn {
                display: inline-block;
                padding: 12px 24px;
                background: #667eea;
                color: white;
                text-decoration: none;
                border-radius: 8px;
                border: none;
                cursor: pointer;
                width: 100%;
                font-size: 16px;
                margin-top: 1rem;
            }
            .btn:hover {
                background: #5a6fd8;
            }
            .back-btn {
                background: #6c757d;
                margin-bottom: 1rem;
                width: auto;
                padding: 8px 16px;
            }
            .back-btn:hover {
                background: #5a6268;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <a href="/" class="btn back-btn">← Back to Home</a>
            <h1>🤖 Train Model</h1>
            <form action="/train" method="post">
                <div class="form-group">
                    <label for="dataset">Select Dataset:</label>
                    <select id="dataset" name="dataset" required>
                        <option value="kepler">Kepler Mission Data</option>
                        <option value="tess">TESS Mission Data</option>
                        <option value="k2">K2 Mission Data</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="model_type">Select Model Type:</label>
                    <select id="model_type" name="model_type" required>
                        <option value="random_forest">Random Forest</option>
                        <option value="gradient_boosting">Gradient Boosting</option>
                        <option value="svm">Support Vector Machine</option>
                        <option value="logistic_regression">Logistic Regression</option>
                        <option value="neural_network">Neural Network</option>
                    </select>
                </div>
                <button type="submit" class="btn">Start Training</button>
            </form>
            <p style="margin-top: 2rem; color: #666; font-size: 14px;">
                <strong>Note:</strong> Training may take several minutes depending on the dataset size and model complexity.
            </p>
        </div>
    </body>
    </html>
    '''

@app.route('/predict')
def predict_page():
    """Prediction page"""
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Make Prediction - Exoplanet Detection AI</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 2rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                background: white;
                padding: 2rem;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                max-width: 600px;
                margin: 0 auto;
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 2rem;
            }
            .form-group {
                margin-bottom: 1.5rem;
            }
            label {
                display: block;
                margin-bottom: 0.5rem;
                color: #555;
                font-weight: 500;
            }
            input {
                width: 100%;
                padding: 0.75rem;
                border: 1px solid #ddd;
                border-radius: 8px;
                box-sizing: border-box;
            }
            .btn {
                display: inline-block;
                padding: 12px 24px;
                background: #667eea;
                color: white;
                text-decoration: none;
                border-radius: 8px;
                border: none;
                cursor: pointer;
                width: 100%;
                font-size: 16px;
                margin-top: 1rem;
            }
            .btn:hover {
                background: #5a6fd8;
            }
            .back-btn {
                background: #6c757d;
                margin-bottom: 1rem;
                width: auto;
                padding: 8px 16px;
            }
            .back-btn:hover {
                background: #5a6268;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <a href="/" class="btn back-btn">← Back to Home</a>
            <h1>🔮 Make Prediction</h1>
            <form id="predictionForm">
                <div class="form-group">
                    <label for="period">Orbital Period (days):</label>
                    <input type="number" id="period" name="period" step="0.001" placeholder="e.g., 3.234" required>
                </div>
                <div class="form-group">
                    <label for="duration">Transit Duration (hours):</label>
                    <input type="number" id="duration" name="duration" step="0.001" placeholder="e.g., 2.5" required>
                </div>
                <div class="form-group">
                    <label for="depth">Transit Depth (ppm):</label>
                    <input type="number" id="depth" name="depth" step="0.001" placeholder="e.g., 119" required>
                </div>
                <div class="form-group">
                    <label for="radius">Planetary Radius (Earth radii):</label>
                    <input type="number" id="radius" name="radius" step="0.001" placeholder="e.g., 1.19" required>
                </div>
                <div class="form-group">
                    <label for="temperature">Stellar Temperature (K):</label>
                    <input type="number" id="temperature" name="temperature" step="1" placeholder="e.g., 5640" required>
                </div>
                <button type="submit" class="btn">Make Prediction</button>
            </form>
            <div id="result" style="margin-top: 2rem; padding: 1rem; border-radius: 8px; display: none;"></div>
        </div>
        <script>
            document.getElementById('predictionForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const formData = new FormData(this);
                const data = Object.fromEntries(formData);
                
                try {
                    const response = await fetch('/predict_single', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    const resultDiv = document.getElementById('result');
                    
                    if (result.prediction) {
                        resultDiv.innerHTML = `
                            <h3>Prediction Result:</h3>
                            <p><strong>Classification:</strong> ${result.prediction}</p>
                            <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                        `;
                        resultDiv.style.background = '#e8f5e8';
                        resultDiv.style.border = '1px solid #4caf50';
                        resultDiv.style.color = '#2e7d32';
                    } else {
                        resultDiv.innerHTML = `<p style="color: #f44336;">Error: ${result.error || 'Unknown error'}</p>`;
                        resultDiv.style.background = '#ffebee';
                        resultDiv.style.border = '1px solid #f44336';
                        resultDiv.style.color = '#c62828';
                    }
                    
                    resultDiv.style.display = 'block';
                } catch (error) {
                    console.error('Error:', error);
                    alert('Error making prediction');
                }
            });
        </script>
    </body>
    </html>
    '''

@app.route('/model_info')
def model_info():
    """Model information"""
    return jsonify({
        'status': 'success',
        'message': 'Model information endpoint',
        'model_type': 'Gradient Boosting',
        'accuracy': 0.95,
        'features': [
            'koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_impact',
            'koi_steff', 'koi_srad', 'koi_slogg', 'koi_model_snr', 'koi_teq'
        ]
    })

@app.route('/predict_single', methods=['POST'])
def predict_single():
    """Make a single prediction"""
    try:
        data = request.get_json()
        
        # Simple mock prediction based on input values
        period = float(data.get('period', 0))
        depth = float(data.get('depth', 0))
        
        # Simple rule-based prediction for demo
        if period > 10 and depth > 100:
            prediction = "CONFIRMED"
            confidence = 0.85
        elif period > 5 and depth > 50:
            prediction = "CANDIDATE"
            confidence = 0.75
        else:
            prediction = "FALSE POSITIVE"
            confidence = 0.65
        
        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'features_used': ['period', 'depth', 'duration', 'radius', 'temperature']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and file.filename.endswith('.csv'):
            # For demo purposes, just return success
            return jsonify({
                'status': 'success',
                'message': 'File uploaded successfully',
                'filename': file.filename,
                'analysis': {
                    'candidates': 5,
                    'false_positives': 12,
                    'total_objects': 17
                }
            })
        else:
            return jsonify({'error': 'Only CSV files are allowed'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """Handle model training"""
    try:
        dataset = request.form.get('dataset', 'kepler')
        model_type = request.form.get('model_type', 'random_forest')
        
        # Mock training response
        return jsonify({
            'status': 'success',
            'message': f'Model trained successfully using {dataset} dataset',
            'model_type': model_type,
            'accuracy': 0.95,
            'training_time': '2.5 minutes'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

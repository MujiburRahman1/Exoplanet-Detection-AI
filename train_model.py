#!/usr/bin/env python3
"""
Training script for exoplanet detection models.
This script can be run from the command line to train models on the NASA datasets.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.data_loader import ExoplanetDataLoader
from src.models.exoplanet_classifier import ExoplanetClassifier
import joblib
import json

def main():
    parser = argparse.ArgumentParser(description='Train exoplanet detection models')
    parser.add_argument('--dataset', choices=['kepler', 'tess', 'k2'], default='kepler',
                       help='Dataset to use for training')
    parser.add_argument('--model', choices=['random_forest', 'gradient_boosting', 'svm', 
                                          'logistic_regression', 'neural_network'], 
                       default='random_forest', help='Model type to train')
    parser.add_argument('--output', default='models/trained_model.joblib',
                       help='Output path for trained model')
    parser.add_argument('--hyperparameter-tuning', action='store_true',
                       help='Enable hyperparameter tuning')
    parser.add_argument('--data-dir', default='data',
                       help='Directory containing the datasets')
    
    args = parser.parse_args()
    
    print(f"Training {args.model} model on {args.dataset} dataset...")
    
    # Initialize data loader
    data_loader = ExoplanetDataLoader(data_dir=args.data_dir)
    
    # Load dataset
    if args.dataset == 'kepler':
        df = data_loader.load_kepler_data()
        target_column = 'koi_pdisposition'  # Disposition Using Kepler Data
    elif args.dataset == 'tess':
        df = data_loader.load_tess_data()
        target_column = 'tfopwg_disposition'  # TESS Follow-up Observing Program Working Group Disposition
    elif args.dataset == 'k2':
        df = data_loader.load_k2_data()
        target_column = 'disposition'  # Archive Disposition
    
    if df.empty:
        print(f"Error: {args.dataset} dataset not found or empty")
        print("Please ensure the dataset files are in the data/ directory")
        return 1
    
    print(f"Loaded dataset with {len(df)} records and {len(df.columns)} features")
    
    # Preprocess data
    X, y = data_loader.preprocess_data(df, target_column)
    
    if y is None:
        print(f"Error: Target column '{target_column}' not found in dataset")
        return 1
    
    print(f"Preprocessed data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Initialize classifier
    classifier = ExoplanetClassifier(model_type=args.model)
    
    # Hyperparameter tuning if requested
    if args.hyperparameter_tuning:
        print("Performing hyperparameter tuning...")
        tuning_results = classifier.hyperparameter_tuning(X, y)
        print(f"Best parameters: {tuning_results['best_params']}")
        print(f"Best score: {tuning_results['best_score']:.4f}")
    
    # Train the model
    print("Training model...")
    results = classifier.train(X, y)
    
    print(f"Training completed!")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Cross-validation score: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save the model
    classifier.save_model(args.output)
    print(f"Model saved to {args.output}")
    
    # Save training results
    results_file = args.output.replace('.joblib', '_results.json')
    with open(results_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for key, value in results.items():
            if hasattr(value, 'tolist'):  # numpy arrays
                json_results[key] = value.tolist()
            elif isinstance(value, dict):
                json_results[key] = {k: (v.tolist() if hasattr(v, 'tolist') else v) 
                                   for k, v in value.items()}
            else:
                json_results[key] = value
        
        json.dump(json_results, f, indent=2)
    
    print(f"Training results saved to {results_file}")
    
    return 0

if __name__ == '__main__':
    exit(main())

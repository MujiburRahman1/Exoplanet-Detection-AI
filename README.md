# Exoplanet Detection AI/ML Project

An artificial intelligence and machine learning system for automatically identifying exoplanets using NASA's open-source datasets from Kepler, K2, and TESS missions.

## Project Overview

This project aims to create an automated system for exoplanet detection that can:
- Analyze transit data from space-based missions
- Classify objects as confirmed exoplanets, planetary candidates, or false positives
- Provide a web interface for researchers and enthusiasts
- Allow data upload and model retraining

## Features

- **Multi-dataset Support**: Works with Kepler, K2, and TESS datasets
- **Machine Learning Models**: Multiple ML algorithms for classification
- **Web Interface**: User-friendly interface for data analysis
- **Real-time Analysis**: Upload and analyze new data
- **Model Statistics**: View accuracy metrics and performance
- **Hyperparameter Tuning**: Adjust model parameters through the interface

## Project Structure

```
exoplanet-detection/
├── data/                   # Raw and processed datasets
├── notebooks/              # Jupyter notebooks for exploration
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── models/            # ML model implementations
│   ├── web/               # Web interface
│   └── utils/             # Utility functions
├── models/                 # Trained model files
├── static/                 # Web assets (CSS, JS, images)
├── templates/              # HTML templates
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

### Quick Setup
1. Run the setup script: `python setup.py`
2. Download NASA datasets to the `data/` directory (see data/README.md)
3. Run the web application: `python src/web/app.py`

### Manual Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Create necessary directories (already created by setup script)
3. Download NASA datasets to the `data/` directory
4. Run the web application: `python src/web/app.py`

## Datasets

- **Kepler Objects of Interest (KOI)**: Confirmed exoplanets, candidates, and false positives
- **TESS Objects of Interest (TOI)**: TESS mission data with classifications
- **K2 Planets and Candidates**: K2 mission transit data

## Usage

### Web Interface
1. Start the web application: `python src/web/app.py`
2. Open your browser to `http://localhost:5000`
3. Upload data files or train models through the interface

### Command Line Training
```bash
# Train a Random Forest model on Kepler data
python train_model.py --dataset kepler --model random_forest

# Train with hyperparameter tuning
python train_model.py --dataset tess --model gradient_boosting --hyperparameter-tuning

# See all options
python train_model.py --help
```

### Data Exploration
1. Use Jupyter notebooks in the `notebooks/` directory
2. Start Jupyter: `jupyter notebook`
3. Open `notebooks/data_exploration.ipynb`

### API Endpoints
- `GET /` - Main interface
- `POST /upload` - Upload data files
- `POST /train` - Train models
- `POST /analyze` - Analyze uploaded data
- `POST /predict_single` - Make single predictions

## Contributing

This project is open for contributions. Please read the contributing guidelines before submitting pull requests.

## License

This project is licensed under the MIT License.

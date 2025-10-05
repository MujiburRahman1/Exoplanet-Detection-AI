"""
Data loading and preprocessing utilities for exoplanet datasets.
Handles Kepler, K2, and TESS datasets from NASA.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExoplanetDataLoader:
    """Load and preprocess exoplanet datasets from NASA missions."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing the dataset files
        """
        self.data_dir = data_dir
        self.datasets = {}
        
    def load_kepler_data(self, filename: str = None) -> pd.DataFrame:
        """
        Load Kepler Objects of Interest dataset.
        
        Args:
            filename: Name of the Kepler dataset file (if None, auto-detect)
            
        Returns:
            DataFrame containing Kepler data
        """
        if filename is None:
            # Auto-detect Kepler file
            kepler_files = [f for f in os.listdir(self.data_dir) if 'koi' in f.lower() and f.endswith('.csv')]
            if kepler_files:
                filename = kepler_files[0]  # Use the first found file
            else:
                filename = "kepler_objects_of_interest.csv"  # Default fallback
        
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"Kepler dataset not found at {filepath}")
            return pd.DataFrame()
            
        try:
            # NASA CSV files have metadata headers, need to find the actual data start
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Find the line that starts with the column names (usually after # comments)
            header_line = None
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('#') and ',' in line:
                    header_line = i
                    break
            
            if header_line is None:
                logger.error(f"Could not find data header in {filepath}")
                return pd.DataFrame()
            
            # Read the CSV starting from the header line
            df = pd.read_csv(filepath, skiprows=header_line)
            logger.info(f"Loaded Kepler dataset with {len(df)} records")
            
            # Standardize column names
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            
            # Store in datasets dict
            self.datasets['kepler'] = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading Kepler dataset: {e}")
            return pd.DataFrame()
    
    def load_tess_data(self, filename: str = None) -> pd.DataFrame:
        """
        Load TESS Objects of Interest dataset.
        
        Args:
            filename: Name of the TESS dataset file (if None, auto-detect)
            
        Returns:
            DataFrame containing TESS data
        """
        if filename is None:
            # Auto-detect TESS file
            tess_files = [f for f in os.listdir(self.data_dir) if 'toi' in f.lower() and f.endswith('.csv')]
            if tess_files:
                filename = tess_files[0]  # Use the first found file
            else:
                filename = "tess_objects_of_interest.csv"  # Default fallback
        
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"TESS dataset not found at {filepath}")
            return pd.DataFrame()
            
        try:
            # NASA CSV files have metadata headers, need to find the actual data start
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Find the line that starts with the column names (usually after # comments)
            header_line = None
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('#') and ',' in line:
                    header_line = i
                    break
            
            if header_line is None:
                logger.error(f"Could not find data header in {filepath}")
                return pd.DataFrame()
            
            # Read the CSV starting from the header line
            df = pd.read_csv(filepath, skiprows=header_line)
            logger.info(f"Loaded TESS dataset with {len(df)} records")
            
            # Standardize column names
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            
            # Store in datasets dict
            self.datasets['tess'] = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading TESS dataset: {e}")
            return pd.DataFrame()
    
    def load_k2_data(self, filename: str = None) -> pd.DataFrame:
        """
        Load K2 Planets and Candidates dataset.
        
        Args:
            filename: Name of the K2 dataset file (if None, auto-detect)
            
        Returns:
            DataFrame containing K2 data
        """
        if filename is None:
            # Auto-detect K2 file
            k2_files = [f for f in os.listdir(self.data_dir) if 'k2' in f.lower() and f.endswith('.csv')]
            if k2_files:
                filename = k2_files[0]  # Use the first found file
            else:
                filename = "k2_planets_and_candidates.csv"  # Default fallback
        
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"K2 dataset not found at {filepath}")
            return pd.DataFrame()
            
        try:
            # NASA CSV files have metadata headers, need to find the actual data start
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Find the line that starts with the column names (usually after # comments)
            header_line = None
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('#') and ',' in line:
                    header_line = i
                    break
            
            if header_line is None:
                logger.error(f"Could not find data header in {filepath}")
                return pd.DataFrame()
            
            # Read the CSV starting from the header line
            df = pd.read_csv(filepath, skiprows=header_line)
            logger.info(f"Loaded K2 dataset with {len(df)} records")
            
            # Standardize column names
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            
            # Store in datasets dict
            self.datasets['k2'] = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading K2 dataset: {e}")
            return pd.DataFrame()
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available datasets.
        
        Returns:
            Dictionary containing all loaded datasets
        """
        self.load_kepler_data()
        self.load_tess_data()
        self.load_k2_data()
        
        return self.datasets
    
    def get_common_features(self) -> List[str]:
        """
        Identify common features across all datasets.
        
        Returns:
            List of common feature names
        """
        if not self.datasets:
            self.load_all_datasets()
        
        common_features = set()
        for i, (name, df) in enumerate(self.datasets.items()):
            if i == 0:
                common_features = set(df.columns)
            else:
                common_features = common_features.intersection(set(df.columns))
        
        return list(common_features)
    
    def preprocess_data(self, df: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Preprocess the dataset for machine learning.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column for classification
            
        Returns:
            Tuple of (features_df, target_series)
        """
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Handle missing values
        processed_df = self._handle_missing_values(processed_df)
        
        # Remove non-numeric columns that aren't useful for ML
        processed_df = self._remove_non_numeric_columns(processed_df)
        
        # Extract target if specified
        target = None
        if target_column and target_column in processed_df.columns:
            target = processed_df[target_column]
            processed_df = processed_df.drop(columns=[target_column])
        
        # Remove any remaining non-numeric columns
        processed_df = processed_df.select_dtypes(include=[np.number])
        
        logger.info(f"Preprocessed data shape: {processed_df.shape}")
        
        return processed_df, target
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Fill numeric columns with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Fill categorical columns with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        return df
    
    def _remove_non_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove columns that are not useful for machine learning."""
        # Remove ID columns and text descriptions
        columns_to_remove = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['id', 'name', 'description', 'url', 'link']):
                columns_to_remove.append(col)
        
        df = df.drop(columns=columns_to_remove)
        return df
    
    def get_dataset_info(self) -> Dict[str, Dict]:
        """
        Get information about loaded datasets.
        
        Returns:
            Dictionary containing dataset information
        """
        info = {}
        for name, df in self.datasets.items():
            info[name] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'dtypes': df.dtypes.to_dict()
            }
        return info

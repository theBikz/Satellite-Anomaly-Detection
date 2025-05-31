#!/usr/bin/env python3
"""
Satellite Orbit Anomaly Detection using XGBoost

This script implements an XGBoost-based anomaly detection system for satellite orbit data.
It detects maneuvers by forecasting orbital elements and identifying large prediction residuals.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
import glob
from typing import Dict, List, Tuple, Optional
import re

# Machine Learning imports
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import IsolationForest

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SatelliteAnomalyDetector:
    """
    XGBoost-based anomaly detection system for satellite orbit data.
    """
    
    def __init__(self, data_dir: str = '.'):
        """
        Initialize the anomaly detector.
        
        Args:
            data_dir: Directory containing orbital_elements and manoeuvres folders
        """
        self.data_dir = data_dir
        self.orbital_data = {}
        self.maneuver_data = {}
        self.models = {}
        self.scalers = {}
        self.results = {}
        
        # XGBoost parameters
        self.xgb_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
    def load_orbital_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load orbital elements data from CSV files.
        
        Returns:
            Dictionary mapping satellite names to DataFrames
        """
        orbital_dir = os.path.join(self.data_dir, 'orbital_elements')
        csv_files = glob.glob(os.path.join(orbital_dir, '*.csv'))
        
        print(f"Loading orbital data from {len(csv_files)} files...")
        
        for file_path in csv_files:
            satellite_name = os.path.basename(file_path).replace('.csv', '')
            
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                # Clean column names
                df.columns = df.columns.str.strip()
                
                # Sort by timestamp
                df = df.sort_index()
                
                # Remove any duplicate timestamps
                df = df[~df.index.duplicated(keep='first')]
                
                self.orbital_data[satellite_name] = df
                print(f"  Loaded {satellite_name}: {len(df)} records from {df.index.min()} to {df.index.max()}")
                
            except Exception as e:
                print(f"  Error loading {satellite_name}: {e}")
                
        return self.orbital_data
    
    def parse_maneuver_timestamp(self, year: int, day_of_year: int, hour: int, minute: int) -> datetime:
        """
        Parse maneuver timestamp from year, day of year, hour, minute format.
        
        Args:
            year: Year
            day_of_year: Day of year (1-366)
            hour: Hour (0-23)
            minute: Minute (0-59)
            
        Returns:
            datetime object
        """
        base_date = datetime(year, 1, 1)
        target_date = base_date + timedelta(days=day_of_year - 1, hours=hour, minutes=minute)
        return target_date
    
    def parse_fengyun_timestamp(self, timestamp_str: str) -> datetime:
        """
        Parse Fengyun timestamp format.
        
        Args:
            timestamp_str: Timestamp string in format "YYYY-MM-DDTHH:MM:SS CST"
            
        Returns:
            datetime object
        """
        # Remove CST and parse
        clean_str = timestamp_str.replace(' CST', '').strip('"')
        return datetime.fromisoformat(clean_str)
    
    def load_maneuver_data(self) -> Dict[str, List[datetime]]:
        """
        Load maneuver data from text files.
        
        Returns:
            Dictionary mapping satellite names to lists of maneuver timestamps
        """
        maneuver_dir = os.path.join(self.data_dir, 'manoeuvres')
        maneuver_files = glob.glob(os.path.join(maneuver_dir, '*man*.txt*'))
        
        print(f"Loading maneuver data from {len(maneuver_files)} files...")
        
        for file_path in maneuver_files:
            filename = os.path.basename(file_path)
            
            # Determine satellite name from filename
            if 'topman' in filename:
                satellite_name = 'TOPEX'
            elif 'ja1man' in filename:
                satellite_name = 'Jason-1'
            elif 'ja2man' in filename:
                satellite_name = 'Jason-2'
            elif 'ja3man' in filename:
                satellite_name = 'Jason-3'
            elif 'cs2man' in filename:
                satellite_name = 'CryoSat-2'
            elif 'en1man' in filename:
                satellite_name = 'Envisat-1'
            elif 'srlman' in filename:
                satellite_name = 'SARAL'
            elif 's6aman' in filename:
                satellite_name = 'Sentinel-6A'
            elif 's3aman' in filename:
                satellite_name = 'Sentinel-3A'
            elif 's3bman' in filename:
                satellite_name = 'Sentinel-3B'
            elif 'sp2man' in filename:
                satellite_name = 'SPOT-2'
            elif 'sp4man' in filename:
                satellite_name = 'SPOT-4'
            elif 'sp5man' in filename:
                satellite_name = 'SPOT-5'
            elif 'h2aman' in filename:
                satellite_name = 'Haiyang-2A'
            elif 'h2cman' in filename:
                satellite_name = 'Haiyang-2C'
            elif 'h2dman' in filename:
                satellite_name = 'Haiyang-2D'
            elif 'FY' in filename:
                # Fengyun satellites
                if 'FY2D' in filename:
                    satellite_name = 'Fengyun-2D'
                elif 'FY2E' in filename:
                    satellite_name = 'Fengyun-2E'
                elif 'FY2F' in filename:
                    satellite_name = 'Fengyun-2F'
                elif 'FY2H' in filename:
                    satellite_name = 'Fengyun-2H'
                elif 'FY4A' in filename:
                    satellite_name = 'Fengyun-4A'
                else:
                    satellite_name = filename.replace('.txt', '').replace('.fy', '')
            else:
                satellite_name = filename.replace('man.txt', '').replace('.txt', '')
            
            maneuver_times = []
            
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    if filename.endswith('.fy'):
                        # Fengyun format
                        parts = line.split('"')
                        if len(parts) >= 3:
                            start_time_str = parts[1]
                            maneuver_time = self.parse_fengyun_timestamp(start_time_str)
                            maneuver_times.append(maneuver_time)
                    else:
                        # Standard format
                        parts = line.split()
                        if len(parts) >= 8:
                            satellite_id = parts[0]
                            year = int(parts[1])
                            day_of_year = int(parts[2])
                            hour = int(parts[3])
                            minute = int(parts[4])
                            
                            maneuver_time = self.parse_maneuver_timestamp(year, day_of_year, hour, minute)
                            maneuver_times.append(maneuver_time)
                
                self.maneuver_data[satellite_name] = sorted(maneuver_times)
                print(f"  Loaded {satellite_name}: {len(maneuver_times)} maneuvers")
                
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
        
        return self.maneuver_data
    
    def create_features(self, df: pd.DataFrame, target_col: str, lookback: int = 5) -> pd.DataFrame:
        """
        Create features for time series forecasting.
        
        Args:
            df: DataFrame with orbital elements
            target_col: Column to predict
            lookback: Number of previous timesteps to use as features
            
        Returns:
            DataFrame with features
        """
        feature_df = df.copy()
        
        # Time-based features
        feature_df['hour'] = feature_df.index.hour
        feature_df['day_of_year'] = feature_df.index.dayofyear
        feature_df['month'] = feature_df.index.month
        feature_df['year'] = feature_df.index.year
        
        # Lag features
        for i in range(1, lookback + 1):
            feature_df[f'{target_col}_lag_{i}'] = feature_df[target_col].shift(i)
        
        # Rolling statistics
        for window in [3, 7, 14]:
            feature_df[f'{target_col}_rolling_mean_{window}'] = feature_df[target_col].rolling(window=window).mean()
            feature_df[f'{target_col}_rolling_std_{window}'] = feature_df[target_col].rolling(window=window).std()
        
        # Differences
        feature_df[f'{target_col}_diff_1'] = feature_df[target_col].diff(1)
        feature_df[f'{target_col}_diff_2'] = feature_df[target_col].diff(2)
        
        # Rate of change
        feature_df[f'{target_col}_pct_change'] = feature_df[target_col].pct_change()
        
        # Drop rows with NaN values
        feature_df = feature_df.dropna()
        
        return feature_df
    
    def train_xgboost_model(self, satellite_name: str, target_col: str, 
                           test_size: float = 0.2, lookback: int = 5) -> Dict:
        """
        Train XGBoost model for a specific satellite and orbital element.
        
        Args:
            satellite_name: Name of the satellite
            target_col: Orbital element to predict
            test_size: Fraction of data to use for testing
            lookback: Number of previous timesteps to use as features
            
        Returns:
            Dictionary with model, scaler, and metrics
        """
        if satellite_name not in self.orbital_data:
            raise ValueError(f"No orbital data found for {satellite_name}")
        
        df = self.orbital_data[satellite_name].copy()
        
        if target_col not in df.columns:
            raise ValueError(f"Column {target_col} not found in {satellite_name} data")
        
        print(f"Training XGBoost model for {satellite_name} - {target_col}")
        
        # Create features
        feature_df = self.create_features(df, target_col, lookback)
        
        # Prepare features and target
        feature_cols = [col for col in feature_df.columns if col != target_col]
        X = feature_df[feature_cols]
        y = feature_df[target_col]
        
        # Split data chronologically
        split_idx = int(len(feature_df) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train XGBoost model
        model = xgb.XGBRegressor(**self.xgb_params)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Calculate residuals
        train_residuals = y_train - y_train_pred
        test_residuals = y_test - y_test_pred
        
        # Store results
        result = {
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'train_residuals': train_residuals,
            'test_residuals': test_residuals,
            'train_timestamps': feature_df.index[:split_idx],
            'test_timestamps': feature_df.index[split_idx:],
            'metrics': {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2
            }
        }
        
        # Store in class
        if satellite_name not in self.models:
            self.models[satellite_name] = {}
        self.models[satellite_name][target_col] = result
        
        print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        print(f"  Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")
        
        return result
    
    def detect_anomalies(self, satellite_name: str, target_col: str, 
                        threshold_method: str = 'std', threshold_factor: float = 3.0) -> pd.DataFrame:
        """
        Detect anomalies based on prediction residuals.
        
        Args:
            satellite_name: Name of the satellite
            target_col: Orbital element that was predicted
            threshold_method: Method to determine threshold ('std', 'quantile', 'isolation_forest')
            threshold_factor: Factor for threshold calculation
            
        Returns:
            DataFrame with anomaly detection results
        """
        if satellite_name not in self.models or target_col not in self.models[satellite_name]:
            raise ValueError(f"No trained model found for {satellite_name} - {target_col}")
        
        result = self.models[satellite_name][target_col]
        
        # Combine all residuals and timestamps
        all_residuals = np.concatenate([result['train_residuals'], result['test_residuals']])
        all_timestamps = np.concatenate([result['train_timestamps'], result['test_timestamps']])
        all_actual = np.concatenate([result['y_train'], result['y_test']])
        all_predicted = np.concatenate([result['y_train_pred'], result['y_test_pred']])
        
        # Calculate absolute residuals
        abs_residuals = np.abs(all_residuals)
        
        # Determine threshold
        if threshold_method == 'std':
            threshold = np.mean(abs_residuals) + threshold_factor * np.std(abs_residuals)
        elif threshold_method == 'quantile':
            threshold = np.quantile(abs_residuals, 1 - threshold_factor / 100)
        elif threshold_method == 'isolation_forest':
            iso_forest = IsolationForest(contamination=threshold_factor / 100, random_state=42)
            anomaly_labels = iso_forest.fit_predict(abs_residuals.reshape(-1, 1))
            threshold = np.min(abs_residuals[anomaly_labels == -1]) if np.any(anomaly_labels == -1) else np.inf
        else:
            raise ValueError(f"Unknown threshold method: {threshold_method}")
        
        # Detect anomalies
        is_anomaly = abs_residuals > threshold
        
        # Create results DataFrame
        anomaly_df = pd.DataFrame({
            'timestamp': all_timestamps,
            'actual': all_actual,
            'predicted': all_predicted,
            'residual': all_residuals,
            'abs_residual': abs_residuals,
            'is_anomaly': is_anomaly,
            'threshold': threshold
        })
        
        anomaly_df.set_index('timestamp', inplace=True)
        anomaly_df = anomaly_df.sort_index()
        
        print(f"Detected {np.sum(is_anomaly)} anomalies out of {len(all_residuals)} predictions")
        print(f"Anomaly rate: {np.sum(is_anomaly) / len(all_residuals) * 100:.2f}%")
        print(f"Threshold: {threshold:.6f}")
        
        return anomaly_df
    
    def evaluate_detection_performance(self, satellite_name: str, target_col: str, 
                                     anomaly_df: pd.DataFrame, tolerance_hours: int = 24) -> Dict:
        """
        Evaluate anomaly detection performance against ground truth maneuvers.
        
        Args:
            satellite_name: Name of the satellite
            target_col: Orbital element that was predicted
            anomaly_df: DataFrame with anomaly detection results
            tolerance_hours: Hours of tolerance for matching detected anomalies to maneuvers
            
        Returns:
            Dictionary with performance metrics
        """
        if satellite_name not in self.maneuver_data:
            print(f"No ground truth maneuver data available for {satellite_name}")
            return {}
        
        maneuver_times = self.maneuver_data[satellite_name]
        detected_anomalies = anomaly_df[anomaly_df['is_anomaly']].index
        
        # Convert to pandas datetime if needed
        maneuver_times = pd.to_datetime(maneuver_times)
        detected_anomalies = pd.to_datetime(detected_anomalies)
        
        # Filter maneuvers to the time range of our data
        data_start = anomaly_df.index.min()
        data_end = anomaly_df.index.max()
        relevant_maneuvers = maneuver_times[(maneuver_times >= data_start) & (maneuver_times <= data_end)]
        
        print(f"Evaluating detection performance for {satellite_name} - {target_col}")
        print(f"  Relevant maneuvers in data range: {len(relevant_maneuvers)}")
        print(f"  Detected anomalies: {len(detected_anomalies)}")
        
        if len(relevant_maneuvers) == 0:
            print("  No maneuvers in data range for evaluation")
            return {}
        
        # Match detected anomalies to maneuvers
        tolerance = pd.Timedelta(hours=tolerance_hours)
        true_positives = 0
        matched_maneuvers = set()
        
        for anomaly_time in detected_anomalies:
            for i, maneuver_time in enumerate(relevant_maneuvers):
                if abs(anomaly_time - maneuver_time) <= tolerance:
                    if i not in matched_maneuvers:
                        true_positives += 1
                        matched_maneuvers.add(i)
                    break
        
        false_positives = len(detected_anomalies) - true_positives
        false_negatives = len(relevant_maneuvers) - true_positives
        
        # Calculate metrics
        precision = true_positives / len(detected_anomalies) if len(detected_anomalies) > 0 else 0
        recall = true_positives / len(relevant_maneuvers) if len(relevant_maneuvers) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        performance = {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'total_maneuvers': len(relevant_maneuvers),
            'total_detections': len(detected_anomalies)
        }
        
        print(f"  True Positives: {true_positives}")
        print(f"  False Positives: {false_positives}")
        print(f"  False Negatives: {false_negatives}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1_score:.3f}")
        
        return performance
    
    def plot_results(self, satellite_name: str, target_col: str, 
                    anomaly_df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Plot the results of anomaly detection.
        
        Args:
            satellite_name: Name of the satellite
            target_col: Orbital element that was predicted
            anomaly_df: DataFrame with anomaly detection results
            save_path: Path to save the plot (optional)
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Actual vs Predicted values
        axes[0].plot(anomaly_df.index, anomaly_df['actual'], label='Actual', alpha=0.7)
        axes[0].plot(anomaly_df.index, anomaly_df['predicted'], label='Predicted', alpha=0.7)
        axes[0].set_title(f'{satellite_name} - {target_col}: Actual vs Predicted')
        axes[0].set_ylabel(target_col)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Residuals with anomalies highlighted
        axes[1].plot(anomaly_df.index, anomaly_df['residual'], label='Residuals', alpha=0.7)
        axes[1].axhline(y=anomaly_df['threshold'].iloc[0], color='red', linestyle='--', label='Threshold')
        axes[1].axhline(y=-anomaly_df['threshold'].iloc[0], color='red', linestyle='--')
        
        # Highlight anomalies
        anomaly_points = anomaly_df[anomaly_df['is_anomaly']]
        if len(anomaly_points) > 0:
            axes[1].scatter(anomaly_points.index, anomaly_points['residual'], 
                          color='red', s=50, label='Detected Anomalies', zorder=5)
        
        axes[1].set_title(f'{satellite_name} - {target_col}: Prediction Residuals')
        axes[1].set_ylabel('Residual')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Absolute residuals
        axes[2].plot(anomaly_df.index, anomaly_df['abs_residual'], label='Absolute Residuals', alpha=0.7)
        axes[2].axhline(y=anomaly_df['threshold'].iloc[0], color='red', linestyle='--', label='Threshold')
        
        if len(anomaly_points) > 0:
            axes[2].scatter(anomaly_points.index, anomaly_points['abs_residual'], 
                          color='red', s=50, label='Detected Anomalies', zorder=5)
        
        # Add ground truth maneuvers if available
        if satellite_name in self.maneuver_data:
            maneuver_times = pd.to_datetime(self.maneuver_data[satellite_name])
            data_start = anomaly_df.index.min()
            data_end = anomaly_df.index.max()
            relevant_maneuvers = maneuver_times[(maneuver_times >= data_start) & (maneuver_times <= data_end)]
            
            for ax in axes:
                for maneuver_time in relevant_maneuvers:
                    ax.axvline(x=maneuver_time, color='green', linestyle=':', alpha=0.7, linewidth=2)
            
            # Add legend entry for maneuvers
            axes[2].axvline(x=relevant_maneuvers[0] if len(relevant_maneuvers) > 0 else anomaly_df.index[0], 
                          color='green', linestyle=':', alpha=0.7, linewidth=2, label='Ground Truth Maneuvers')
        
        axes[2].set_title(f'{satellite_name} - {target_col}: Absolute Residuals')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Absolute Residual')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def run_full_analysis(self, satellite_name: str, target_col: str = 'eccentricity', 
                         threshold_method: str = 'std', threshold_factor: float = 3.0,
                         save_plots: bool = True) -> Dict:
        """
        Run the complete anomaly detection analysis for a satellite.
        
        Args:
            satellite_name: Name of the satellite to analyze
            target_col: Orbital element to predict and analyze
            threshold_method: Method to determine anomaly threshold
            threshold_factor: Factor for threshold calculation
            save_plots: Whether to save plots
            
        Returns:
            Dictionary with all results
        """
        print(f"\n{'='*60}")
        print(f"Running full analysis for {satellite_name}")
        print(f"{'='*60}")
        
        # Train model
        model_result = self.train_xgboost_model(satellite_name, target_col)
        
        # Detect anomalies
        anomaly_df = self.detect_anomalies(satellite_name, target_col, 
                                         threshold_method, threshold_factor)
        
        # Evaluate performance
        performance = self.evaluate_detection_performance(satellite_name, target_col, anomaly_df)
        
        # Plot results
        if save_plots:
            plot_path = f"{satellite_name}_{target_col}_anomaly_detection.png"
            self.plot_results(satellite_name, target_col, anomaly_df, plot_path)
        else:
            self.plot_results(satellite_name, target_col, anomaly_df)
        
        # Compile results
        full_results = {
            'satellite_name': satellite_name,
            'target_col': target_col,
            'model_metrics': model_result['metrics'],
            'anomaly_detection': anomaly_df,
            'performance_metrics': performance,
            'threshold_method': threshold_method,
            'threshold_factor': threshold_factor
        }
        
        # Store results
        if satellite_name not in self.results:
            self.results[satellite_name] = {}
        self.results[satellite_name][target_col] = full_results
        
        return full_results

def main():
    """
    Main function to run the satellite anomaly detection analysis.
    """
    print("Satellite Orbit Anomaly Detection using XGBoost")
    print("=" * 50)
    
    # Initialize detector
    detector = SatelliteAnomalyDetector()
    
    # Load data
    print("\n1. Loading Data...")
    orbital_data = detector.load_orbital_data()
    maneuver_data = detector.load_maneuver_data()
    
    if not orbital_data:
        print("No orbital data loaded. Please check the data directory structure.")
        return
    
    # Display available satellites
    print(f"\nAvailable satellites for analysis:")
    for satellite in orbital_data.keys():
        print(f"  - {satellite}")
    
    # Run analysis for a few key satellites
    satellites_to_analyze = ['TOPEX', 'Jason-1', 'Jason-2']
    orbital_elements = ['eccentricity', 'inclination', 'mean anomaly']
    
    print(f"\n2. Running Analysis...")
    
    for satellite in satellites_to_analyze:
        if satellite in orbital_data:
            for element in orbital_elements:
                if element in orbital_data[satellite].columns:
                    try:
                        print(f"\nAnalyzing {satellite} - {element}...")
                        results = detector.run_full_analysis(
                            satellite_name=satellite,
                            target_col=element,
                            threshold_method='std',
                            threshold_factor=3.0,
                            save_plots=True
                        )
                    except Exception as e:
                        print(f"Error analyzing {satellite} - {element}: {e}")
                else:
                    print(f"Column {element} not found in {satellite} data")
        else:
            print(f"No data available for {satellite}")
    
    # Summary report
    print(f"\n3. Summary Report")
    print("=" * 50)
    
    for satellite_name, satellite_results in detector.results.items():
        print(f"\n{satellite_name}:")
        for element, results in satellite_results.items():
            metrics = results['model_metrics']
            performance = results['performance_metrics']
            
            print(f"  {element}:")
            print(f"    Model R²: {metrics['test_r2']:.4f}")
            print(f"    Model MSE: {metrics['test_mse']:.6f}")
            
            if performance:
                print(f"    Detection Precision: {performance['precision']:.3f}")
                print(f"    Detection Recall: {performance['recall']:.3f}")
                print(f"    Detection F1-Score: {performance['f1_score']:.3f}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 
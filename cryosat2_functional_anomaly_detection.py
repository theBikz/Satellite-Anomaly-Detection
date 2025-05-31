#!/usr/bin/env python3
"""
CryoSat-2 Orbit Anomaly Detection System - Functional Implementation

Function-based anomaly detection system specifically optimized for CryoSat-2 satellite.
This script identifies orbital maneuvers by forecasting orbital elements and detecting large prediction residuals.

CryoSat-2 Mission Details:
- Launch: April 8, 2010
- Mission: Ice monitoring and Earth observation
- Orbit: Sun-synchronous polar orbit
- Altitude: ~717 km
- Inclination: ~92°

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

# Global configuration for CryoSat-2
CRYOSAT2_CONFIG = {
    'satellite_name': 'CryoSat-2',
    'lookback_days': 7,  # Week lookback for ice mission patterns
    'rolling_windows': [3, 7, 14, 21],  # Including 3-week window for seasonal patterns
    'xgb_params': {
        'objective': 'reg:squarederror',
        'max_depth': 7,  # Slightly deeper for complex orbital patterns
        'learning_rate': 0.08,  # Lower learning rate for better precision
        'n_estimators': 150,  # More estimators for ice mission complexity
        'subsample': 0.85,
        'colsample_bytree': 0.9,
        'min_child_weight': 3,  # Helps with CryoSat-2's data characteristics
        'gamma': 0.1,  # Regularization for orbital stability
        'random_state': 42,
        'n_jobs': -1
    }
}

def initialize_cryosat2_system():
    """
    Initialize the CryoSat-2 anomaly detection system.
    
    Returns:
        Dict: Configuration and initial state for CryoSat-2 analysis
    """
    print(f"CryoSat-2 Anomaly Detection System Initialized")
    print(f"Mission: Earth Ice Monitoring (2010-present)")
    print(f"Orbital characteristics optimized for polar orbit analysis")
    
    return {
        'satellite_name': CRYOSAT2_CONFIG['satellite_name'],
        'orbital_data': None,
        'maneuver_data': [],
        'models': {},
        'scalers': {},
        'results': {},
        'xgb_params': CRYOSAT2_CONFIG['xgb_params'].copy(),
        'lookback_days': CRYOSAT2_CONFIG['lookback_days'],
        'rolling_windows': CRYOSAT2_CONFIG['rolling_windows']
    }

def load_cryosat2_data(data_dir: str = '.') -> Tuple[pd.DataFrame, List[datetime], bool]:
    """
    Load CryoSat-2 specific orbital and maneuver data.
    
    Args:
        data_dir: Directory containing orbital_elements and manoeuvres folders
        
    Returns:
        Tuple of (orbital_data, maneuver_data, success_flag)
    """
    print(f"Loading CryoSat-2 data...")
    
    orbital_data = None
    maneuver_data = []
    
    # Load orbital data
    orbital_file = os.path.join(data_dir, 'orbital_elements', 'CryoSat-2.csv')
    if not os.path.exists(orbital_file):
        print(f"Error: CryoSat-2 orbital data not found at {orbital_file}")
        return None, [], False
    
    try:
        orbital_data = pd.read_csv(orbital_file, index_col=0, parse_dates=True)
        orbital_data.columns = orbital_data.columns.str.strip()
        orbital_data = orbital_data.sort_index()
        orbital_data = orbital_data[~orbital_data.index.duplicated(keep='first')]
        
        print(f"  ✓ Loaded orbital data: {len(orbital_data)} records")
        print(f"  ✓ Time range: {orbital_data.index.min()} to {orbital_data.index.max()}")
        print(f"  ✓ Available elements: {list(orbital_data.columns)}")
        
    except Exception as e:
        print(f"Error loading orbital data: {e}")
        return None, [], False
    
    # Load maneuver data
    maneuver_file = os.path.join(data_dir, 'manoeuvres', 'cs2man.txt')
    if os.path.exists(maneuver_file):
        try:
            with open(maneuver_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 8:
                    year = int(parts[1])
                    day_of_year = int(parts[2])
                    hour = int(parts[3])
                    minute = int(parts[4])
                    
                    base_date = datetime(year, 1, 1)
                    maneuver_time = base_date + timedelta(days=day_of_year - 1, hours=hour, minutes=minute)
                    maneuver_data.append(maneuver_time)
            
            maneuver_data = sorted(maneuver_data)
            print(f"  ✓ Loaded maneuver data: {len(maneuver_data)} maneuvers")
            
            if maneuver_data:
                print(f"  ✓ Maneuver time range: {min(maneuver_data)} to {max(maneuver_data)}")
            
        except Exception as e:
            print(f"Warning: Error loading maneuver data: {e}")
            print(f"  Continuing without ground truth validation")
    else:
        print(f"  Warning: Maneuver file not found at {maneuver_file}")
        print(f"  Continuing without ground truth validation")
    
    return orbital_data, maneuver_data, True

def create_cryosat2_features(df: pd.DataFrame, target_col: str, 
                           lookback_days: int, rolling_windows: List[int]) -> pd.DataFrame:
    """
    Create CryoSat-2 specific features for time series forecasting.
    
    Optimized for polar orbit characteristics and ice mission patterns.
    
    Args:
        df: DataFrame with orbital elements
        target_col: Column to predict
        lookback_days: Number of days to look back for lag features
        rolling_windows: List of rolling window sizes
        
    Returns:
        DataFrame with CryoSat-2 optimized features
    """
    feature_df = df.copy()
    
    # Time-based features (important for polar orbit and seasonal ice patterns)
    feature_df['hour'] = feature_df.index.hour
    feature_df['day_of_year'] = feature_df.index.dayofyear
    feature_df['month'] = feature_df.index.month
    feature_df['year'] = feature_df.index.year
    feature_df['quarter'] = feature_df.index.quarter
    
    # Seasonal features for ice mission (Arctic/Antarctic cycles)
    feature_df['sin_day_of_year'] = np.sin(2 * np.pi * feature_df['day_of_year'] / 365.25)
    feature_df['cos_day_of_year'] = np.cos(2 * np.pi * feature_df['day_of_year'] / 365.25)
    
    # Binary features for Arctic/Antarctic seasons
    feature_df['arctic_summer'] = ((feature_df['day_of_year'] >= 80) & 
                                 (feature_df['day_of_year'] <= 266)).astype(int)
    feature_df['antarctic_summer'] = ((feature_df['day_of_year'] >= 266) | 
                                    (feature_df['day_of_year'] <= 80)).astype(int)
    
    # Lag features optimized for CryoSat-2 orbital period
    for i in range(1, lookback_days + 1):
        feature_df[f'{target_col}_lag_{i}'] = feature_df[target_col].shift(i)
    
    # Rolling statistics with CryoSat-2 specific windows
    for window in rolling_windows:
        feature_df[f'{target_col}_rolling_mean_{window}'] = feature_df[target_col].rolling(window=window).mean()
        feature_df[f'{target_col}_rolling_std_{window}'] = feature_df[target_col].rolling(window=window).std()
        feature_df[f'{target_col}_rolling_min_{window}'] = feature_df[target_col].rolling(window=window).min()
        feature_df[f'{target_col}_rolling_max_{window}'] = feature_df[target_col].rolling(window=window).max()
        feature_df[f'{target_col}_rolling_range_{window}'] = (feature_df[f'{target_col}_rolling_max_{window}'] - 
                                                            feature_df[f'{target_col}_rolling_min_{window}'])
    
    # Difference features
    feature_df[f'{target_col}_diff_1'] = feature_df[target_col].diff(1)
    feature_df[f'{target_col}_diff_2'] = feature_df[target_col].diff(2)
    feature_df[f'{target_col}_diff_7'] = feature_df[target_col].diff(7)  # Weekly difference
    
    # Rate of change features
    feature_df[f'{target_col}_pct_change_1'] = feature_df[target_col].pct_change(1)
    feature_df[f'{target_col}_pct_change_7'] = feature_df[target_col].pct_change(7)
    
    # Volatility measures (important for maneuver detection)
    for window in [7, 14]:
        feature_df[f'{target_col}_volatility_{window}'] = (feature_df[target_col]
                                                         .rolling(window=window)
                                                         .std() / feature_df[target_col]
                                                         .rolling(window=window)
                                                         .mean())
    
    # CryoSat-2 specific orbital features
    if 'inclination' in feature_df.columns and target_col != 'inclination':
        feature_df['inclination_stability'] = feature_df['inclination'].rolling(window=14).std()
    
    # Drop rows with NaN values
    initial_length = len(feature_df)
    feature_df = feature_df.dropna()
    dropped_rows = initial_length - len(feature_df)
    
    print(f"  Feature engineering complete: {len(feature_df.columns)} features created")
    print(f"  Dropped {dropped_rows} rows with NaN values")
    
    return feature_df

def train_cryosat2_model(orbital_data: pd.DataFrame, target_col: str, 
                       xgb_params: Dict, lookback_days: int, rolling_windows: List[int],
                       test_size: float = 0.2) -> Dict:
    """
    Train XGBoost model optimized for CryoSat-2 orbital characteristics.
    
    Args:
        orbital_data: DataFrame with orbital elements
        target_col: Orbital element to predict
        xgb_params: XGBoost parameters
        lookback_days: Number of days for lag features
        rolling_windows: List of rolling window sizes
        test_size: Fraction of data to use for testing
        
    Returns:
        Dictionary with model, scaler, and metrics
    """
    if orbital_data is None:
        raise ValueError("No orbital data provided")
    
    if target_col not in orbital_data.columns:
        raise ValueError(f"Column {target_col} not found in CryoSat-2 data")
    
    print(f"\nTraining CryoSat-2 XGBoost model for {target_col}")
    print("-" * 50)
    
    # Create CryoSat-2 specific features
    feature_df = create_cryosat2_features(orbital_data, target_col, lookback_days, rolling_windows)
    
    # Prepare features and target
    feature_cols = [col for col in feature_df.columns if col != target_col]
    X = feature_df[feature_cols]
    y = feature_df[target_col]
    
    print(f"Training data shape: {X.shape}")
    print(f"Target statistics: mean={y.mean():.6f}, std={y.std():.6f}")
    
    # Split data chronologically (important for time series)
    split_idx = int(len(feature_df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost model with CryoSat-2 optimized parameters
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate comprehensive metrics
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
    
    print(f"\nModel Performance:")
    print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    print(f"  Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")
    print(f"  Train MAE: {train_mae:.6f}, Test MAE: {test_mae:.6f}")
    
    # Feature importance analysis
    feature_importance = model.feature_importances_
    top_features = sorted(zip(feature_cols, feature_importance), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop 10 Important Features for {target_col}:")
    for feature, importance in top_features:
        print(f"  {feature}: {importance:.4f}")
    
    return result

def detect_cryosat2_anomalies(model_result: Dict, target_col: str, 
                            threshold_method: str = 'std', 
                            threshold_factor: float = 3.0) -> pd.DataFrame:
    """
    Detect anomalies in CryoSat-2 orbital data based on prediction residuals.
    
    Args:
        model_result: Results from training function
        target_col: Orbital element that was predicted
        threshold_method: Method to determine threshold ('std', 'quantile', 'isolation_forest')
        threshold_factor: Factor for threshold calculation
        
    Returns:
        DataFrame with anomaly detection results
    """
    print(f"\nDetecting CryoSat-2 anomalies for {target_col}")
    print("-" * 50)
    
    # Combine all residuals and timestamps
    all_residuals = np.concatenate([model_result['train_residuals'], model_result['test_residuals']])
    all_timestamps = np.concatenate([model_result['train_timestamps'], model_result['test_timestamps']])
    all_actual = np.concatenate([model_result['y_train'], model_result['y_test']])
    all_predicted = np.concatenate([model_result['y_train_pred'], model_result['y_test_pred']])
    
    # Calculate absolute residuals
    abs_residuals = np.abs(all_residuals)
    
    # CryoSat-2 specific threshold determination
    if threshold_method == 'std':
        threshold = np.mean(abs_residuals) + threshold_factor * np.std(abs_residuals)
    elif threshold_method == 'quantile':
        threshold = np.quantile(abs_residuals, 1 - threshold_factor / 100)
    elif threshold_method == 'isolation_forest':
        # Optimized for CryoSat-2's maneuver patterns
        iso_forest = IsolationForest(
            contamination=threshold_factor / 100, 
            random_state=42,
            n_estimators=150  # More estimators for better CryoSat-2 detection
        )
        anomaly_labels = iso_forest.fit_predict(abs_residuals.reshape(-1, 1))
        threshold = np.min(abs_residuals[anomaly_labels == -1]) if np.any(anomaly_labels == -1) else np.inf
    elif threshold_method == 'adaptive':
        # CryoSat-2 specific adaptive threshold
        rolling_mean = pd.Series(abs_residuals).rolling(window=30, center=True).mean()
        rolling_std = pd.Series(abs_residuals).rolling(window=30, center=True).std()
        threshold = rolling_mean + threshold_factor * rolling_std
        threshold = threshold.fillna(threshold.mean()).values
    else:
        raise ValueError(f"Unknown threshold method: {threshold_method}")
    
    # Detect anomalies
    if isinstance(threshold, np.ndarray):
        is_anomaly = abs_residuals > threshold
    else:
        is_anomaly = abs_residuals > threshold
        threshold = np.full_like(abs_residuals, threshold)
    
    # Create results DataFrame
    anomaly_df = pd.DataFrame({
        'timestamp': all_timestamps,
        'actual': all_actual,
        'predicted': all_predicted,
        'residual': all_residuals,
        'abs_residual': abs_residuals,
        'is_anomaly': is_anomaly,
        'threshold': threshold if isinstance(threshold, np.ndarray) else np.full_like(abs_residuals, threshold)
    })
    
    anomaly_df.set_index('timestamp', inplace=True)
    anomaly_df = anomaly_df.sort_index()
    
    # CryoSat-2 specific anomaly analysis
    total_anomalies = np.sum(is_anomaly)
    anomaly_rate = total_anomalies / len(all_residuals) * 100
    
    print(f"Anomaly Detection Results:")
    print(f"  Total predictions: {len(all_residuals)}")
    print(f"  Detected anomalies: {total_anomalies}")
    print(f"  Anomaly rate: {anomaly_rate:.2f}%")
    if not isinstance(threshold, np.ndarray):
        print(f"  Threshold: {threshold:.6f}")
    else:
        print(f"  Threshold range: {threshold.min():.6f} to {threshold.max():.6f}")
    
    return anomaly_df

def evaluate_cryosat2_performance(anomaly_df: pd.DataFrame, maneuver_data: List[datetime], 
                                target_col: str, tolerance_hours: int = 24) -> Dict:
    """
    Evaluate CryoSat-2 anomaly detection performance against ground truth maneuvers.
    
    Args:
        anomaly_df: DataFrame with anomaly detection results
        maneuver_data: List of ground truth maneuver times
        target_col: Orbital element that was predicted
        tolerance_hours: Hours of tolerance for matching detected anomalies to maneuvers
        
    Returns:
        Dictionary with performance metrics
    """
    if not maneuver_data:
        print("No ground truth maneuver data available for CryoSat-2")
        return {}
    
    detected_anomalies = anomaly_df[anomaly_df['is_anomaly']].index
    maneuver_times = pd.to_datetime(maneuver_data)
    detected_anomalies = pd.to_datetime(detected_anomalies)
    
    # Filter maneuvers to the time range of our data
    data_start = anomaly_df.index.min()
    data_end = anomaly_df.index.max()
    relevant_maneuvers = maneuver_times[(maneuver_times >= data_start) & (maneuver_times <= data_end)]
    
    print(f"\nCryoSat-2 Detection Performance Evaluation for {target_col}")
    print("-" * 50)
    print(f"Data range: {data_start.date()} to {data_end.date()}")
    print(f"Relevant maneuvers in range: {len(relevant_maneuvers)}")
    print(f"Detected anomalies: {len(detected_anomalies)}")
    
    if len(relevant_maneuvers) == 0:
        print("No maneuvers in data range for evaluation")
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
    
    # Calculate detection efficiency
    detection_rate = true_positives / len(relevant_maneuvers) if len(relevant_maneuvers) > 0 else 0
    false_alarm_rate = false_positives / len(detected_anomalies) if len(detected_anomalies) > 0 else 0
    
    performance = {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'detection_rate': detection_rate,
        'false_alarm_rate': false_alarm_rate,
        'total_maneuvers': len(relevant_maneuvers),
        'total_detections': len(detected_anomalies),
        'tolerance_hours': tolerance_hours
    }
    
    print(f"\nPerformance Metrics:")
    print(f"  True Positives: {true_positives}")
    print(f"  False Positives: {false_positives}")
    print(f"  False Negatives: {false_negatives}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1-Score: {f1_score:.3f}")
    print(f"  Detection Rate: {detection_rate:.3f}")
    print(f"  False Alarm Rate: {false_alarm_rate:.3f}")
    
    return performance

def plot_cryosat2_results(anomaly_df: pd.DataFrame, maneuver_data: List[datetime], 
                        target_col: str, save_path: Optional[str] = None):
    """
    Plot CryoSat-2 specific anomaly detection results.
    
    Args:
        anomaly_df: DataFrame with anomaly detection results
        maneuver_data: List of ground truth maneuver times
        target_col: Orbital element that was predicted
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))
    
    # Plot 1: Actual vs Predicted values
    axes[0].plot(anomaly_df.index, anomaly_df['actual'], label='Actual', alpha=0.8, linewidth=1)
    axes[0].plot(anomaly_df.index, anomaly_df['predicted'], label='Predicted', alpha=0.8, linewidth=1)
    axes[0].set_title(f'CryoSat-2 - {target_col}: Actual vs Predicted Values\n'
                     f'Ice Monitoring Mission (2010-present)', fontsize=14, pad=20)
    axes[0].set_ylabel(target_col)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals with anomalies highlighted
    axes[1].plot(anomaly_df.index, anomaly_df['residual'], label='Residuals', alpha=0.7, linewidth=1)
    
    # Plot threshold
    if len(anomaly_df['threshold'].unique()) == 1:
        threshold_val = anomaly_df['threshold'].iloc[0]
        axes[1].axhline(y=threshold_val, color='red', linestyle='--', 
                      label=f'Threshold: ±{threshold_val:.6f}', alpha=0.8)
        axes[1].axhline(y=-threshold_val, color='red', linestyle='--', alpha=0.8)
    
    # Highlight anomalies
    anomaly_points = anomaly_df[anomaly_df['is_anomaly']]
    if len(anomaly_points) > 0:
        axes[1].scatter(anomaly_points.index, anomaly_points['residual'], 
                      color='red', s=30, label=f'Detected Anomalies ({len(anomaly_points)})', 
                      zorder=5, alpha=0.8)
    
    axes[1].set_title(f'CryoSat-2 - {target_col}: Prediction Residuals')
    axes[1].set_ylabel('Residual')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Absolute residuals
    axes[2].plot(anomaly_df.index, anomaly_df['abs_residual'], 
                label='Absolute Residuals', alpha=0.7, linewidth=1, color='purple')
    
    if len(anomaly_df['threshold'].unique()) == 1:
        axes[2].axhline(y=anomaly_df['threshold'].iloc[0], color='red', 
                      linestyle='--', label='Detection Threshold', alpha=0.8)
    
    if len(anomaly_points) > 0:
        axes[2].scatter(anomaly_points.index, anomaly_points['abs_residual'], 
                      color='red', s=30, label='Detected Anomalies', zorder=5, alpha=0.8)
    
    axes[2].set_title(f'CryoSat-2 - {target_col}: Absolute Residuals')
    axes[2].set_ylabel('Absolute Residual')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Seasonal pattern analysis (CryoSat-2 specific)
    monthly_anomalies = anomaly_df.groupby(anomaly_df.index.month)['is_anomaly'].sum()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    bars = axes[3].bar(range(1, 13), monthly_anomalies.reindex(range(1, 13), fill_value=0), 
                      color='skyblue', alpha=0.7, edgecolor='navy')
    axes[3].set_title(f'CryoSat-2 - {target_col}: Monthly Anomaly Distribution\n'
                     f'(Arctic Summer: Apr-Sep, Antarctic Summer: Oct-Mar)')
    axes[3].set_xlabel('Month')
    axes[3].set_ylabel('Number of Anomalies')
    axes[3].set_xticks(range(1, 13))
    axes[3].set_xticklabels(months)
    axes[3].grid(True, alpha=0.3, axis='y')
    
    # Highlight Arctic and Antarctic seasons
    axes[3].axvspan(4, 9, alpha=0.1, color='red', label='Arctic Summer')
    axes[3].axvspan(10, 12, alpha=0.1, color='blue', label='Antarctic Summer')
    axes[3].axvspan(1, 3, alpha=0.1, color='blue')
    axes[3].legend()
    
    # Add ground truth maneuvers if available
    if maneuver_data:
        maneuver_times = pd.to_datetime(maneuver_data)
        data_start = anomaly_df.index.min()
        data_end = anomaly_df.index.max()
        relevant_maneuvers = maneuver_times[(maneuver_times >= data_start) & (maneuver_times <= data_end)]
        
        for i, ax in enumerate(axes[:3]):  # Don't add to monthly plot
            for maneuver_time in relevant_maneuvers:
                ax.axvline(x=maneuver_time, color='green', linestyle=':', 
                         alpha=0.7, linewidth=2)
        
        if len(relevant_maneuvers) > 0:
            axes[2].axvline(x=relevant_maneuvers[0], color='green', linestyle=':', 
                          alpha=0.7, linewidth=2, label=f'Ground Truth Maneuvers ({len(relevant_maneuvers)})')
            axes[2].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"CryoSat-2 plot saved to {save_path}")
    
    plt.show()

def print_cryosat2_summary(results: Dict):
    """
    Print a comprehensive summary of CryoSat-2 analysis results.
    
    Args:
        results: Results dictionary from run_full_cryosat2_analysis
    """
    print("\n" + "="*60)
    print("CRYOSAT-2 ANALYSIS SUMMARY REPORT")
    print("="*60)
    
    print(f"\nMission Information:")
    print(f"  Satellite: {results['satellite_name']}")
    print(f"  Target Element: {results['target_col']}")
    print(f"  Data Range: {results['data_summary']['time_range']}")
    print(f"  Total Records: {results['data_summary']['total_records']:,}")
    print(f"  Ground Truth Maneuvers: {results['data_summary']['total_maneuvers']}")
    
    metrics = results['model_metrics']
    print(f"\nModel Performance:")
    print(f"  R² Score (Test): {metrics['test_r2']:.4f}")
    print(f"  MSE (Test): {metrics['test_mse']:.6f}")
    print(f"  MAE (Test): {metrics['test_mae']:.6f}")
    print(f"  Model Quality: {'Excellent' if metrics['test_r2'] > 0.8 else 'Good' if metrics['test_r2'] > 0.6 else 'Fair'}")
    
    anomaly_df = results['anomaly_detection']
    total_anomalies = anomaly_df['is_anomaly'].sum()
    anomaly_rate = total_anomalies / len(anomaly_df) * 100
    
    print(f"\nAnomaly Detection:")
    print(f"  Method: {results['threshold_method']}")
    print(f"  Threshold Factor: {results['threshold_factor']}")
    print(f"  Total Predictions: {len(anomaly_df):,}")
    print(f"  Detected Anomalies: {total_anomalies}")
    print(f"  Anomaly Rate: {anomaly_rate:.2f}%")
    
    if results['performance_metrics']:
        perf = results['performance_metrics']
        print(f"\nDetection Performance (vs Ground Truth):")
        print(f"  Precision: {perf['precision']:.3f}")
        print(f"  Recall: {perf['recall']:.3f}")
        print(f"  F1-Score: {perf['f1_score']:.3f}")
        print(f"  Detection Rate: {perf['detection_rate']:.3f}")
        print(f"  False Alarm Rate: {perf['false_alarm_rate']:.3f}")
        print(f"  Performance Grade: {'Excellent' if perf['f1_score'] > 0.8 else 'Good' if perf['f1_score'] > 0.6 else 'Fair'}")
    
    print(f"\nCryoSat-2 Operational Insights:")
    if 'eccentricity' in results['target_col'].lower():
        print(f"  • Eccentricity variations may indicate orbit maintenance maneuvers")
        print(f"  • Critical for maintaining ice measurement accuracy")
    elif 'inclination' in results['target_col'].lower():
        print(f"  • Inclination changes affect polar coverage")
        print(f"  • Important for Arctic/Antarctic monitoring consistency")
    
    print(f"\n{'='*60}")
    print("CryoSat-2 analysis completed successfully!")
    print("="*60)

def run_full_cryosat2_analysis(data_dir: str = '.', target_col: str = 'eccentricity', 
                              threshold_method: str = 'std', threshold_factor: float = 3.0,
                              save_plots: bool = True) -> Dict:
    """
    Run the complete CryoSat-2 anomaly detection analysis.
    
    Args:
        data_dir: Directory containing data files
        target_col: Orbital element to predict and analyze
        threshold_method: Method to determine anomaly threshold
        threshold_factor: Factor for threshold calculation
        save_plots: Whether to save plots
        
    Returns:
        Dictionary with all results
    """
    print("="*60)
    print("CryoSat-2 Orbit Anomaly Detection Analysis")
    print("="*60)
    print("Mission: Earth Ice Monitoring")
    print("Launch: April 8, 2010")
    print("Orbit: Sun-synchronous polar orbit (~717 km)")
    print("="*60)
    
    # Initialize system
    config = initialize_cryosat2_system()
    
    # Load data
    orbital_data, maneuver_data, success = load_cryosat2_data(data_dir)
    if not success:
        raise RuntimeError("Failed to load CryoSat-2 data")
    
    # Train model
    model_result = train_cryosat2_model(
        orbital_data, target_col, config['xgb_params'], 
        config['lookback_days'], config['rolling_windows']
    )
    
    # Detect anomalies
    anomaly_df = detect_cryosat2_anomalies(model_result, target_col, threshold_method, threshold_factor)
    
    # Evaluate performance
    performance = evaluate_cryosat2_performance(anomaly_df, maneuver_data, target_col)
    
    # Plot results
    if save_plots:
        plot_path = f"CryoSat-2_functional_{target_col}_anomaly_detection.png"
        plot_cryosat2_results(anomaly_df, maneuver_data, target_col, plot_path)
    else:
        plot_cryosat2_results(anomaly_df, maneuver_data, target_col)
    
    # Compile comprehensive results
    full_results = {
        'satellite_name': config['satellite_name'],
        'target_col': target_col,
        'model_metrics': model_result['metrics'],
        'anomaly_detection': anomaly_df,
        'performance_metrics': performance,
        'threshold_method': threshold_method,
        'threshold_factor': threshold_factor,
        'data_summary': {
            'total_records': len(orbital_data),
            'time_range': f"{orbital_data.index.min()} to {orbital_data.index.max()}",
            'available_elements': list(orbital_data.columns),
            'total_maneuvers': len(maneuver_data)
        }
    }
    
    # Print comprehensive summary
    print_cryosat2_summary(full_results)
    
    return full_results

def main():
    """
    Main function to run CryoSat-2 specific anomaly detection analysis.
    """
    print("CryoSat-2 Orbit Anomaly Detection System - Functional Implementation")
    print("Specialized for Ice Monitoring Mission")
    
    # Available orbital elements for analysis
    elements_to_analyze = ['eccentricity', 'inclination', 'mean anomaly']
    
    # Store all results
    all_results = {}
    
    # Run analysis for each element
    for element in elements_to_analyze:
        try:
            print(f"\n{'='*80}")
            print(f"Analyzing CryoSat-2 {element.upper()}")
            print('='*80)
            
            results = run_full_cryosat2_analysis(
                data_dir='.',
                target_col=element,
                threshold_method='std',
                threshold_factor=3.0,
                save_plots=True
            )
            
            all_results[element] = results
            
        except Exception as e:
            print(f"Error analyzing {element}: {e}")
            continue
    
    print(f"\n{'='*80}")
    print("CryoSat-2 Complete Mission Analysis Summary")
    print('='*80)
    
    # Overall mission summary
    for element, results in all_results.items():
        metrics = results['model_metrics']
        perf = results.get('performance_metrics', {})
        
        print(f"\n{element.upper()}:")
        print(f"  Model R²: {metrics['test_r2']:.4f}")
        if perf:
            print(f"  F1-Score: {perf['f1_score']:.3f}")
            print(f"  Detection Rate: {perf['detection_rate']:.3f}")
    
    print(f"\nCryoSat-2 functional anomaly detection analysis complete!")
    print(f"Mission status: Operational ice monitoring continues")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Example usage of the Satellite Anomaly Detection System

This script demonstrates various ways to use the anomaly detection system
with different configurations and parameters.
"""

from satellite_anomaly_detection import SatelliteAnomalyDetector
import pandas as pd
import numpy as np

def example_basic_usage():
    """
    Example 1: Basic usage with default parameters
    """
    print("="*60)
    print("Example 1: Basic Usage")
    print("="*60)
    
    # Initialize detector
    detector = SatelliteAnomalyDetector(data_dir='.')
    
    # Load data
    print("Loading data...")
    orbital_data = detector.load_orbital_data()
    maneuver_data = detector.load_maneuver_data()
    
    if not orbital_data:
        print("No orbital data found. Please check your data directory.")
        return
    
    # Show available satellites
    print(f"\nAvailable satellites: {list(orbital_data.keys())}")
    
    # Run analysis for TOPEX satellite
    if 'TOPEX' in orbital_data:
        print("\nRunning analysis for TOPEX satellite...")
        results = detector.run_full_analysis(
            satellite_name='TOPEX',
            target_col='eccentricity',
            save_plots=True
        )
        
        print(f"Model R²: {results['model_metrics']['test_r2']:.4f}")
        if results['performance_metrics']:
            print(f"Detection F1-Score: {results['performance_metrics']['f1_score']:.3f}")

def example_custom_parameters():
    """
    Example 2: Custom parameters and multiple orbital elements
    """
    print("\n" + "="*60)
    print("Example 2: Custom Parameters")
    print("="*60)
    
    detector = SatelliteAnomalyDetector(data_dir='.')
    detector.load_orbital_data()
    detector.load_maneuver_data()
    
    # Custom XGBoost parameters
    detector.xgb_params.update({
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 200
    })
    
    # Analyze multiple orbital elements
    satellite = 'TOPEX'
    elements = ['eccentricity', 'inclination']
    
    if satellite in detector.orbital_data:
        for element in elements:
            if element in detector.orbital_data[satellite].columns:
                print(f"\nAnalyzing {satellite} - {element} with custom parameters...")
                
                results = detector.run_full_analysis(
                    satellite_name=satellite,
                    target_col=element,
                    threshold_method='quantile',
                    threshold_factor=5.0,  # 5% quantile
                    save_plots=False
                )

def example_threshold_comparison():
    """
    Example 3: Compare different threshold methods
    """
    print("\n" + "="*60)
    print("Example 3: Threshold Method Comparison")
    print("="*60)
    
    detector = SatelliteAnomalyDetector(data_dir='.')
    detector.load_orbital_data()
    detector.load_maneuver_data()
    
    satellite = 'TOPEX'
    element = 'eccentricity'
    
    if satellite not in detector.orbital_data:
        print(f"No data available for {satellite}")
        return
    
    if element not in detector.orbital_data[satellite].columns:
        print(f"Column {element} not found in {satellite} data")
        return
    
    # Train model once
    print(f"Training model for {satellite} - {element}...")
    detector.train_xgboost_model(satellite, element)
    
    # Compare different threshold methods
    threshold_methods = [
        ('std', 3.0),
        ('quantile', 5.0),
        ('isolation_forest', 10.0)
    ]
    
    results_comparison = {}
    
    for method, factor in threshold_methods:
        print(f"\nTesting {method} threshold method...")
        
        # Detect anomalies
        anomaly_df = detector.detect_anomalies(
            satellite, element, method, factor
        )
        
        # Evaluate performance
        performance = detector.evaluate_detection_performance(
            satellite, element, anomaly_df
        )
        
        results_comparison[method] = {
            'anomaly_count': anomaly_df['is_anomaly'].sum(),
            'performance': performance
        }
    
    # Print comparison
    print("\n" + "-"*50)
    print("Threshold Method Comparison Results:")
    print("-"*50)
    
    for method, results in results_comparison.items():
        print(f"\n{method.upper()}:")
        print(f"  Anomalies detected: {results['anomaly_count']}")
        if results['performance']:
            perf = results['performance']
            print(f"  Precision: {perf['precision']:.3f}")
            print(f"  Recall: {perf['recall']:.3f}")
            print(f"  F1-Score: {perf['f1_score']:.3f}")

def example_batch_analysis():
    """
    Example 4: Batch analysis of multiple satellites
    """
    print("\n" + "="*60)
    print("Example 4: Batch Analysis")
    print("="*60)
    
    detector = SatelliteAnomalyDetector(data_dir='.')
    detector.load_orbital_data()
    detector.load_maneuver_data()
    
    # Define satellites and elements to analyze
    analysis_config = {
        'TOPEX': ['eccentricity', 'inclination'],
        'Jason-1': ['eccentricity'],
        'Jason-2': ['eccentricity']
    }
    
    batch_results = {}
    
    for satellite, elements in analysis_config.items():
        if satellite not in detector.orbital_data:
            print(f"Skipping {satellite} - no data available")
            continue
        
        batch_results[satellite] = {}
        
        for element in elements:
            if element not in detector.orbital_data[satellite].columns:
                print(f"Skipping {satellite}-{element} - column not found")
                continue
            
            try:
                print(f"\nProcessing {satellite} - {element}...")
                
                results = detector.run_full_analysis(
                    satellite_name=satellite,
                    target_col=element,
                    threshold_method='std',
                    threshold_factor=3.0,
                    save_plots=False  # Don't save plots for batch processing
                )
                
                batch_results[satellite][element] = results
                
            except Exception as e:
                print(f"Error processing {satellite}-{element}: {e}")
    
    # Generate summary report
    print("\n" + "="*60)
    print("BATCH ANALYSIS SUMMARY REPORT")
    print("="*60)
    
    for satellite, satellite_results in batch_results.items():
        print(f"\n{satellite}:")
        print("-" * (len(satellite) + 1))
        
        for element, results in satellite_results.items():
            metrics = results['model_metrics']
            performance = results['performance_metrics']
            
            print(f"\n  {element}:")
            print(f"    Model Performance:")
            print(f"      R²: {metrics['test_r2']:.4f}")
            print(f"      MSE: {metrics['test_mse']:.6f}")
            print(f"      MAE: {metrics['test_mae']:.6f}")
            
            if performance:
                print(f"    Detection Performance:")
                print(f"      Precision: {performance['precision']:.3f}")
                print(f"      Recall: {performance['recall']:.3f}")
                print(f"      F1-Score: {performance['f1_score']:.3f}")
                print(f"      True Positives: {performance['true_positives']}")
                print(f"      False Positives: {performance['false_positives']}")
                print(f"      False Negatives: {performance['false_negatives']}")
            else:
                print(f"    Detection Performance: No ground truth available")

def example_data_exploration():
    """
    Example 5: Data exploration and statistics
    """
    print("\n" + "="*60)
    print("Example 5: Data Exploration")
    print("="*60)
    
    detector = SatelliteAnomalyDetector(data_dir='.')
    orbital_data = detector.load_orbital_data()
    maneuver_data = detector.load_maneuver_data()
    
    if not orbital_data:
        print("No data available for exploration")
        return
    
    print("\nORBITAL DATA SUMMARY:")
    print("-" * 30)
    
    for satellite, df in orbital_data.items():
        print(f"\n{satellite}:")
        print(f"  Time range: {df.index.min()} to {df.index.max()}")
        print(f"  Data points: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        
        # Basic statistics for eccentricity
        if 'eccentricity' in df.columns:
            ecc_stats = df['eccentricity'].describe()
            print(f"  Eccentricity stats:")
            print(f"    Mean: {ecc_stats['mean']:.6f}")
            print(f"    Std: {ecc_stats['std']:.6f}")
            print(f"    Min: {ecc_stats['min']:.6f}")
            print(f"    Max: {ecc_stats['max']:.6f}")
    
    print("\nMANEUVER DATA SUMMARY:")
    print("-" * 30)
    
    for satellite, maneuvers in maneuver_data.items():
        if maneuvers:
            maneuver_times = pd.to_datetime(maneuvers)
            print(f"\n{satellite}:")
            print(f"  Total maneuvers: {len(maneuvers)}")
            print(f"  Time range: {maneuver_times.min()} to {maneuver_times.max()}")
            
            # Calculate time between maneuvers
            if len(maneuver_times) > 1:
                time_diffs = maneuver_times.diff().dropna()
                avg_interval = time_diffs.mean()
                print(f"  Average interval: {avg_interval}")

def main():
    """
    Run all examples
    """
    print("Satellite Anomaly Detection - Example Usage")
    print("=" * 50)
    
    try:
        # Run examples
        example_data_exploration()
        example_basic_usage()
        example_custom_parameters()
        example_threshold_comparison()
        example_batch_analysis()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Please check your data directory and file formats.")

if __name__ == "__main__":
    main() 
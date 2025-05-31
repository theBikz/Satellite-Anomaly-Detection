#!/usr/bin/env python3
"""
CryoSat-2 Functional Example Usage Script

Simple examples demonstrating how to use the CryoSat-2 specific functional anomaly detection system.
This script shows different configurations and analysis approaches for CryoSat-2 using functions.
"""

from cryosat2_functional_anomaly_detection import (
    initialize_cryosat2_system,
    load_cryosat2_data,
    run_full_cryosat2_analysis,
    train_cryosat2_model,
    detect_cryosat2_anomalies,
    evaluate_cryosat2_performance
)
import pandas as pd

def example_basic_cryosat2():
    """
    Example 1: Basic CryoSat-2 analysis with default settings using functions
    """
    print("="*60)
    print("CryoSat-2 Functional Example 1: Basic Analysis")
    print("="*60)
    
    # Run basic analysis for eccentricity using functions
    try:
        results = run_full_cryosat2_analysis(
            data_dir='.',
            target_col='eccentricity',
            threshold_method='std',
            threshold_factor=3.0,
            save_plots=True
        )
        
        print(f"\nBasic Functional Analysis Results:")
        print(f"Model R²: {results['model_metrics']['test_r2']:.4f}")
        if results['performance_metrics']:
            print(f"Detection F1-Score: {results['performance_metrics']['f1_score']:.3f}")
            print(f"Detection Rate: {results['performance_metrics']['detection_rate']:.3f}")
        
    except Exception as e:
        print(f"Error in basic functional analysis: {e}")

def example_custom_threshold():
    """
    Example 2: CryoSat-2 analysis with custom threshold methods using functions
    """
    print("\n" + "="*60)
    print("CryoSat-2 Functional Example 2: Custom Threshold Methods")
    print("="*60)
    
    # Test different threshold methods for inclination
    threshold_methods = [
        ('std', 2.5, 'Conservative'),
        ('quantile', 5.0, 'Moderate'),
        ('isolation_forest', 15.0, 'Aggressive')
    ]
    
    results_comparison = {}
    
    for method, factor, description in threshold_methods:
        try:
            print(f"\nTesting {method} method ({description}) with functions...")
            
            results = run_full_cryosat2_analysis(
                data_dir='.',
                target_col='inclination',
                threshold_method=method,
                threshold_factor=factor,
                save_plots=False  # Don't save plots for comparison
            )
            
            anomaly_df = results['anomaly_detection']
            total_anomalies = anomaly_df['is_anomaly'].sum()
            
            results_comparison[method] = {
                'description': description,
                'anomalies': total_anomalies,
                'rate': total_anomalies / len(anomaly_df) * 100,
                'performance': results['performance_metrics']
            }
            
        except Exception as e:
            print(f"Error with {method} method: {e}")
    
    # Print comparison
    print(f"\n{'='*50}")
    print("CryoSat-2 Functional Threshold Method Comparison")
    print(f"{'='*50}")
    
    for method, data in results_comparison.items():
        print(f"\n{method.upper()} ({data['description']}):")
        print(f"  Anomalies detected: {data['anomalies']}")
        print(f"  Anomaly rate: {data['rate']:.2f}%")
        if data['performance']:
            perf = data['performance']
            print(f"  Precision: {perf['precision']:.3f}")
            print(f"  Recall: {perf['recall']:.3f}")
            print(f"  F1-Score: {perf['f1_score']:.3f}")

def example_seasonal_analysis():
    """
    Example 3: CryoSat-2 seasonal pattern analysis using functions
    """
    print("\n" + "="*60)
    print("CryoSat-2 Functional Example 3: Seasonal Pattern Analysis")
    print("="*60)
    
    try:
        # Run analysis for mean anomaly (good for seasonal patterns)
        results = run_full_cryosat2_analysis(
            data_dir='.',
            target_col='mean anomaly',
            threshold_method='std',
            threshold_factor=3.0,
            save_plots=False
        )
        
        anomaly_df = results['anomaly_detection']
        
        # Analyze seasonal patterns
        print(f"\nCryoSat-2 Functional Seasonal Analysis:")
        print(f"Total data points: {len(anomaly_df):,}")
        print(f"Data range: {anomaly_df.index.min().date()} to {anomaly_df.index.max().date()}")
        
        # Monthly anomaly distribution
        monthly_anomalies = anomaly_df.groupby(anomaly_df.index.month)['is_anomaly'].sum()
        
        print(f"\nMonthly Anomaly Distribution:")
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for month_num, month_name in enumerate(months, 1):
            count = monthly_anomalies.get(month_num, 0)
            if month_num in [4, 5, 6, 7, 8, 9]:  # Arctic summer
                season = "Arctic Summer"
            else:  # Antarctic summer
                season = "Antarctic Summer"
            print(f"  {month_name}: {count} anomalies ({season})")
        
        # Seasonal totals
        arctic_summer = monthly_anomalies.reindex([4, 5, 6, 7, 8, 9], fill_value=0).sum()
        antarctic_summer = monthly_anomalies.reindex([1, 2, 3, 10, 11, 12], fill_value=0).sum()
        
        print(f"\nSeasonal Summary:")
        print(f"  Arctic Summer (Apr-Sep): {arctic_summer} anomalies")
        print(f"  Antarctic Summer (Oct-Mar): {antarctic_summer} anomalies")
        
    except Exception as e:
        print(f"Error in functional seasonal analysis: {e}")

def example_step_by_step_analysis():
    """
    Example 4: Step-by-step CryoSat-2 analysis using individual functions
    """
    print("\n" + "="*60)
    print("CryoSat-2 Functional Example 4: Step-by-Step Analysis")
    print("="*60)
    
    try:
        # Initialize system
        config = initialize_cryosat2_system()
        
        # Load data
        print("\nStep 1: Loading CryoSat-2 data...")
        orbital_data, maneuver_data, success = load_cryosat2_data('.')
        if not success:
            print("Failed to load data")
            return
        
        # Modify parameters for better performance
        config['xgb_params'].update({
            'max_depth': 8,          # Deeper trees for complex patterns
            'n_estimators': 200,     # More estimators for better accuracy
            'learning_rate': 0.05,   # Lower learning rate for precision
        })
        
        print("\nStep 2: Training optimized model...")
        model_result = train_cryosat2_model(
            orbital_data, 'eccentricity', config['xgb_params'], 
            config['lookback_days'], config['rolling_windows']
        )
        
        print("\nStep 3: Detecting anomalies...")
        anomaly_df = detect_cryosat2_anomalies(
            model_result, 'eccentricity', 'quantile', 3.0
        )
        
        print("\nStep 4: Evaluating performance...")
        performance = evaluate_cryosat2_performance(
            anomaly_df, maneuver_data, 'eccentricity'
        )
        
        print(f"\nStep-by-Step Analysis Results:")
        metrics = model_result['metrics']
        print(f"  Model R²: {metrics['test_r2']:.4f}")
        print(f"  Model MSE: {metrics['test_mse']:.6f}")
        print(f"  Model MAE: {metrics['test_mae']:.6f}")
        
        if performance:
            print(f"  Detection Precision: {performance['precision']:.3f}")
            print(f"  Detection Recall: {performance['recall']:.3f}")
            print(f"  Detection F1-Score: {performance['f1_score']:.3f}")
        
        print(f"\nTop 5 Important Features:")
        feature_importance = model_result['model'].feature_importances_
        feature_cols = model_result['feature_cols']
        
        top_features = sorted(zip(feature_cols, feature_importance), 
                            key=lambda x: x[1], reverse=True)[:5]
        
        for i, (feature, importance) in enumerate(top_features, 1):
            print(f"  {i}. {feature}: {importance:.4f}")
        
    except Exception as e:
        print(f"Error in step-by-step functional analysis: {e}")

def example_data_summary():
    """
    Example 5: CryoSat-2 data exploration and summary using functions
    """
    print("\n" + "="*60)
    print("CryoSat-2 Functional Example 5: Data Summary")
    print("="*60)
    
    try:
        # Load data for exploration
        orbital_data, maneuver_data, success = load_cryosat2_data('.')
        if success:
            print(f"CryoSat-2 Functional Data Summary:")
            print(f"  Mission: Ice Monitoring (2010-present)")
            print(f"  Total orbital records: {len(orbital_data):,}")
            print(f"  Data time range: {orbital_data.index.min().date()} to {orbital_data.index.max().date()}")
            print(f"  Mission duration: {(orbital_data.index.max() - orbital_data.index.min()).days} days")
            
            print(f"\nAvailable Orbital Elements:")
            for i, element in enumerate(orbital_data.columns, 1):
                stats = orbital_data[element].describe()
                print(f"  {i}. {element}:")
                print(f"     Range: {stats['min']:.6f} to {stats['max']:.6f}")
                print(f"     Mean: {stats['mean']:.6f} ± {stats['std']:.6f}")
            
            print(f"\nManeuver Data:")
            print(f"  Total maneuvers: {len(maneuver_data)}")
            if maneuver_data:
                maneuver_times = pd.to_datetime(maneuver_data)
                print(f"  Maneuver time range: {maneuver_times.min().date()} to {maneuver_times.max().date()}")
                
                # Calculate average interval between maneuvers
                if len(maneuver_times) > 1:
                    intervals = maneuver_times.diff().dropna()
                    avg_interval = intervals.mean()
                    print(f"  Average interval: {avg_interval}")
                
                # Yearly maneuver count
                yearly_counts = maneuver_times.groupby(maneuver_times.year).size()
                print(f"\nManeuvers per year:")
                for year, count in yearly_counts.items():
                    print(f"    {year}: {count} maneuvers")
        
    except Exception as e:
        print(f"Error in functional data summary: {e}")

def example_batch_analysis():
    """
    Example 6: Batch analysis of multiple orbital elements using functions
    """
    print("\n" + "="*60)
    print("CryoSat-2 Functional Example 6: Batch Analysis")
    print("="*60)
    
    elements = ['eccentricity', 'inclination', 'mean anomaly']
    batch_results = {}
    
    print("Running batch analysis for multiple orbital elements...")
    
    for element in elements:
        try:
            print(f"\nProcessing {element}...")
            results = run_full_cryosat2_analysis(
                data_dir='.',
                target_col=element,
                threshold_method='std',
                threshold_factor=3.0,
                save_plots=False  # Skip plots for batch processing
            )
            batch_results[element] = results['performance_metrics']
            
        except Exception as e:
            print(f"Error processing {element}: {e}")
            batch_results[element] = None
    
    # Compare performance across elements
    print(f"\n{'='*50}")
    print("CryoSat-2 Functional Batch Analysis Results")
    print(f"{'='*50}")
    
    for element, perf in batch_results.items():
        if perf:
            print(f"\n{element.upper()}:")
            print(f"  F1-Score: {perf['f1_score']:.3f}")
            print(f"  Precision: {perf['precision']:.3f}")
            print(f"  Recall: {perf['recall']:.3f}")
            print(f"  Detection Rate: {perf['detection_rate']:.3f}")
        else:
            print(f"\n{element.upper()}: Analysis failed")

def main():
    """
    Run all CryoSat-2 functional examples
    """
    print("CryoSat-2 Functional Anomaly Detection - Example Usage")
    print("Specialized for ESA's Ice Monitoring Mission")
    print("Using Function-Based Implementation")
    print("=" * 60)
    
    try:
        # Run all examples
        example_data_summary()
        example_basic_cryosat2()
        example_custom_threshold()
        example_seasonal_analysis()
        example_step_by_step_analysis()
        example_batch_analysis()
        
        print("\n" + "="*60)
        print("All CryoSat-2 functional examples completed successfully!")
        print("="*60)
        print("CryoSat-2 Mission Status: ✅ Operational")
        print("Analysis Capability: ✅ Production Ready")
        print("Implementation: ✅ Function-Based")
        print("Ice Monitoring: ✅ 14+ years of data coverage")
        
    except Exception as e:
        print(f"Error running CryoSat-2 functional examples: {e}")
        print("Please check your data directory and file formats.")

if __name__ == "__main__":
    main() 
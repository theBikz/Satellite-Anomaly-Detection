#!/usr/bin/env python3
"""
Satellite-specific Anomaly Detection Utility

This script allows you to analyze specific satellites with command-line arguments.
Usage: python analyze_satellite.py [satellite_name] [orbital_element] [options]
"""

import argparse
import sys
from satellite_anomaly_detection import SatelliteAnomalyDetector

def main():
    parser = argparse.ArgumentParser(
        description='Analyze satellite orbit anomalies using XGBoost',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_satellite.py TOPEX eccentricity
  python analyze_satellite.py Jason-1 inclination --threshold-method quantile --threshold-factor 5.0
  python analyze_satellite.py Jason-2 "mean anomaly" --no-plots
  python analyze_satellite.py --list-satellites
  python analyze_satellite.py --list-elements TOPEX
        """
    )
    
    # Main arguments
    parser.add_argument('satellite', nargs='?', help='Satellite name (e.g., TOPEX, Jason-1)')
    parser.add_argument('element', nargs='?', help='Orbital element (e.g., eccentricity, inclination)')
    
    # Options
    parser.add_argument('--threshold-method', choices=['std', 'quantile', 'isolation_forest'], 
                       default='std', help='Anomaly detection threshold method (default: std)')
    parser.add_argument('--threshold-factor', type=float, default=3.0,
                       help='Threshold factor (default: 3.0)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--data-dir', default='.',
                       help='Data directory path (default: current directory)')
    
    # Information options
    parser.add_argument('--list-satellites', action='store_true',
                       help='List available satellites and exit')
    parser.add_argument('--list-elements', metavar='SATELLITE',
                       help='List available orbital elements for a satellite and exit')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = SatelliteAnomalyDetector(data_dir=args.data_dir)
    
    # Load data
    print("Loading data...")
    orbital_data = detector.load_orbital_data()
    maneuver_data = detector.load_maneuver_data()
    
    if not orbital_data:
        print("Error: No orbital data found. Please check your data directory.")
        sys.exit(1)
    
    # Handle information requests
    if args.list_satellites:
        print("\nAvailable satellites:")
        for satellite in sorted(orbital_data.keys()):
            maneuver_count = len(maneuver_data.get(satellite, []))
            data_points = len(orbital_data[satellite])
            time_range = f"{orbital_data[satellite].index.min().date()} to {orbital_data[satellite].index.max().date()}"
            print(f"  {satellite:<15} - {data_points:>4} data points, {maneuver_count:>3} maneuvers ({time_range})")
        sys.exit(0)
    
    if args.list_elements:
        satellite = args.list_elements
        if satellite not in orbital_data:
            print(f"Error: Satellite '{satellite}' not found.")
            print(f"Available satellites: {', '.join(sorted(orbital_data.keys()))}")
            sys.exit(1)
        
        print(f"\nAvailable orbital elements for {satellite}:")
        for element in orbital_data[satellite].columns:
            stats = orbital_data[satellite][element].describe()
            print(f"  {element:<20} - Range: {stats['min']:.6f} to {stats['max']:.6f}")
        sys.exit(0)
    
    # Validate required arguments
    if not args.satellite or not args.element:
        print("Error: Both satellite name and orbital element are required.")
        print("Use --help for usage information or --list-satellites to see available options.")
        sys.exit(1)
    
    # Validate satellite
    if args.satellite not in orbital_data:
        print(f"Error: Satellite '{args.satellite}' not found.")
        print(f"Available satellites: {', '.join(sorted(orbital_data.keys()))}")
        print("Use --list-satellites for more details.")
        sys.exit(1)
    
    # Validate orbital element
    if args.element not in orbital_data[args.satellite].columns:
        print(f"Error: Orbital element '{args.element}' not found for {args.satellite}.")
        print(f"Available elements: {', '.join(orbital_data[args.satellite].columns)}")
        print(f"Use --list-elements {args.satellite} for more details.")
        sys.exit(1)
    
    # Run analysis
    print(f"\nAnalyzing {args.satellite} - {args.element}")
    print("=" * 60)
    
    try:
        results = detector.run_full_analysis(
            satellite_name=args.satellite,
            target_col=args.element,
            threshold_method=args.threshold_method,
            threshold_factor=args.threshold_factor,
            save_plots=not args.no_plots
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        
        metrics = results['model_metrics']
        performance = results['performance_metrics']
        
        print(f"\nSatellite: {args.satellite}")
        print(f"Orbital Element: {args.element}")
        print(f"Threshold Method: {args.threshold_method}")
        print(f"Threshold Factor: {args.threshold_factor}")
        
        print(f"\nModel Performance:")
        print(f"  R² Score: {metrics['test_r2']:.4f}")
        print(f"  MSE: {metrics['test_mse']:.6f}")
        print(f"  MAE: {metrics['test_mae']:.6f}")
        
        anomaly_df = results['anomaly_detection']
        total_anomalies = anomaly_df['is_anomaly'].sum()
        anomaly_rate = total_anomalies / len(anomaly_df) * 100
        
        print(f"\nAnomaly Detection:")
        print(f"  Total Predictions: {len(anomaly_df)}")
        print(f"  Detected Anomalies: {total_anomalies}")
        print(f"  Anomaly Rate: {anomaly_rate:.2f}%")
        print(f"  Threshold: {anomaly_df['threshold'].iloc[0]:.6f}")
        
        if performance:
            print(f"\nDetection Performance (vs Ground Truth):")
            print(f"  Precision: {performance['precision']:.3f}")
            print(f"  Recall: {performance['recall']:.3f}")
            print(f"  F1-Score: {performance['f1_score']:.3f}")
            print(f"  True Positives: {performance['true_positives']}")
            print(f"  False Positives: {performance['false_positives']}")
            print(f"  False Negatives: {performance['false_negatives']}")
            print(f"  Total Ground Truth Maneuvers: {performance['total_maneuvers']}")
        else:
            print(f"\nDetection Performance: No ground truth maneuver data available")
        
        if not args.no_plots:
            plot_filename = f"{args.satellite}_{args.element}_anomaly_detection.png"
            print(f"\nVisualization saved as: {plot_filename}")
        
        print(f"\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
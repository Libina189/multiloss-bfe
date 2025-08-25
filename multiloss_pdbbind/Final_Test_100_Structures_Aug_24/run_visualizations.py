#!/usr/bin/env python3
"""
Simple script to run PGGCN visualizations
Save this as 'run_visualizations.py'
"""

from pggcn_visualization import *
import glob
import sys

def main():
    print("PGGCN Visualization Runner")
    print("=" * 40)
    
    # Find all pickle files in current directory
    pickle_files = glob.glob("PGGCN_results_*.pkl")
    
    if not pickle_files:
        print("No PGGCN results pickle files found!")
        print("Looking for files matching: PGGCN_results_*.pkl")
        print("Available files:")
        all_files = glob.glob("*.pkl")
        for f in all_files:
            print(f"  {f}")
        return
    
    print(f"Found {len(pickle_files)} pickle file(s):")
    for i, f in enumerate(pickle_files):
        print(f"  {i+1}. {f}")
    
    # If only one file, use it automatically
    if len(pickle_files) == 1:
        pickle_file = pickle_files[0]
        print(f"\nUsing: {pickle_file}")
    else:
        # Let user choose
        try:
            choice = int(input("\nEnter file number to analyze: ")) - 1
            pickle_file = pickle_files[choice]
        except (ValueError, IndexError):
            print("Invalid choice!")
            return
    
    # Load results
    print(f"\nLoading results from: {pickle_file}")
    results_data = load_results(pickle_file)
    
    if results_data is None:
        print("Failed to load results!")
        return
    
    # Generate all plots
    print("\nGenerating all visualization plots...")
    try:
        plots = generate_all_plots(results_data, save_plots=True)
        print("✓ All plots generated and saved successfully!")
        
        # Create comprehensive report
        print("\nGenerating comprehensive report...")
        report, _ = create_comprehensive_report(results_data, save_report=True)
        print("✓ Comprehensive report generated!")
        
        print(f"\nAnalysis complete! Check the generated files:")
        print("- *.png files for visualizations")
        print("- *_report.txt for comprehensive analysis")
        
    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
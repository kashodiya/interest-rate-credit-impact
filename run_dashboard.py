"""
Script to run the analysis pipeline and launch the dashboard.
"""

import sys
from main import main
from dashboard.dashboard_app import DashboardApp


def run_dashboard():
    """
    Run the complete analysis pipeline and launch the dashboard.
    """
    print("Starting Interest Rate and Consumer Credit Analysis System")
    print("=" * 60)
    
    # Run analysis pipeline
    print("\nRunning analysis pipeline...")
    results = main()
    
    if results is None:
        print("\n✗ Analysis failed. Cannot launch dashboard.")
        print("Please check the errors above and ensure data files are available.")
        sys.exit(1)
    
    # Launch dashboard
    print("\n" + "=" * 60)
    print("Launching dashboard...")
    print("=" * 60)
    
    try:
        dashboard = DashboardApp(results)
        dashboard.run(host="127.0.0.1", port=8050, debug=False)
    except Exception as e:
        print(f"\n✗ Error launching dashboard: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_dashboard()

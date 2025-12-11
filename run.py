"""
Zero - Master Run Script
========================

This script automates the complete startup process for the Zero chatbot:
1. Checks if data files exist (module_map.json, class_schedule.json)
2. If missing or --force flag is provided, runs ingestion (ingest.py)
3. Launches the Gradio web interface (app.py)

Usage:
    python run.py              # Run app (ingest only if data missing)
    python run.py --force      # Force re-ingest data then run app
    python run.py --ingest     # Only run ingestion, don't start app
"""

import sys
import os
import subprocess
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_FOLDER = PROJECT_ROOT / "data"
MODULE_MAP = DATA_FOLDER / "module_map.json"
CLASS_SCHEDULE = DATA_FOLDER / "class_schedule.json"
DB_PATH = PROJECT_ROOT / "chroma_db"

INGEST_SCRIPT = PROJECT_ROOT / "ingest.py"
APP_SCRIPT = PROJECT_ROOT / "app.py"


def check_data_exists():
    """Check if required data files exist."""
    return MODULE_MAP.exists() and CLASS_SCHEDULE.exists()


def run_ingestion():
    """Run the data ingestion pipeline."""
    print("\n" + "="*80)
    print("📥 RUNNING DATA INGESTION")
    print("="*80 + "\n")
    
    result = subprocess.run([sys.executable, str(INGEST_SCRIPT)])
    
    if result.returncode != 0:
        print("\n❌ Ingestion failed!")
        return False
    
    print("\n✅ Ingestion completed successfully!")
    return True


def run_app():
    """Launch the Gradio web application."""
    print("\n" + "="*80)
    print("🚀 LAUNCHING ZERO WEB APPLICATION")
    print("="*80 + "\n")
    
    # This will run the app in the current process (blocking)
    result = subprocess.run([sys.executable, str(APP_SCRIPT)])
    return result.returncode == 0


def main():
    """Main entry point for the run script."""
    
    # Parse arguments
    force_ingest = "--force" in sys.argv or "-f" in sys.argv
    ingest_only = "--ingest" in sys.argv or "-i" in sys.argv
    
    print("="*80)
    print("🤖 ZERO - AI ACADEMIC ADVISOR")
    print("Rhine-Waal University of Applied Sciences")
    print("="*80)
    
    # Determine if we need to run ingestion
    needs_ingest = force_ingest or not check_data_exists()
    
    if needs_ingest:
        if not check_data_exists():
            print("\n⚠️  Data files not found. Running ingestion...")
        elif force_ingest:
            print("\n🔄 Force flag detected. Re-ingesting data...")
        
        if not run_ingestion():
            print("\n❌ Cannot start app without data. Exiting.")
            sys.exit(1)
    else:
        print("\n✅ Data files found. Skipping ingestion.")
        print("   (Use --force to re-ingest data)")
    
    # If --ingest flag was provided, stop here
    if ingest_only:
        print("\n✅ Ingestion complete. Exiting (use 'python run.py' to start app).")
        sys.exit(0)
    
    # Launch the app
    print("\n" + "="*80)
    print("READY TO START")
    print("="*80)
    print("\nThe web interface will open at: http://localhost:7860")
    print("Press Ctrl+C to stop the server.\n")
    
    try:
        run_app()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down Zero. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()

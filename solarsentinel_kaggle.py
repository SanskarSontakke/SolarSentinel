# solarsentinel_kaggle.py
"""
SolarSentinel Kaggle Orchestrator — Verification Only.
Ues this script to verify the performance of submission.py.
DO NOT use to generate submission.py (it is already optimized).
"""

import os
import sys
import subprocess
import shutil

def setup():
    print("Preparing SolarSentinel Verification Environment...")
    os.makedirs("snapshots", exist_ok=True)
    
    # Establish baseline for verification if it doesn't exist
    if not os.path.exists("snapshots/agent_v0.py"):
        if os.path.exists("submission.py"):
            shutil.copy2("submission.py", "snapshots/agent_v0.py")
            print("Current submission established as agent_v0 baseline.")
    
    print("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir", "kaggle-environments>=1.28.0", "numpy"], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print("Verified environments Ready.")

def run():
    print("\n" + "="*60)
    print("  VERIFICATION: Testing SolarSentinel v11.0")
    print("="*60)
    
    if not os.path.exists("submission.py"):
        print("[ERROR] submission.py not found! Ensure the agent file is in the current directory.")
        return

    # Test 1: Agent syntax and import
    result = subprocess.run(
        [sys.executable, "-c", "import submission; print('Agent imports OK')"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[ERROR] Agent failed to import: {result.stderr}")
        return
    print(result.stdout.strip())
    
    # Test 2: Benchmark
    if os.path.exists("benchmark.py"):
        print("\nRunning benchmark vs snapshots/agent_v0.py...")
        subprocess.run(
            [sys.executable, "benchmark.py",
             "--agent-a", "submission.py",
             "--agent-b", "snapshots/agent_v0.py",
             "--games", "10", "--quick"],
            capture_output=False, text=True
        )
    
    print("\n" + "="*60)
    print("  VERIFICATION COMPLETE.")
    print("  UPLOAD:   submission.py  -> To Kaggle Competition Submit.")
    print("  NOTEBOOK: solarsentinel_kaggle.py -> For local testing only.")
    print("="*60)

if __name__ == "__main__":
    setup()
    run()

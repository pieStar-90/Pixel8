import time
import subprocess
from pathlib import Path

WATCH_FILE = Path("data/synthesised_posts.csv")

def run_pipeline():
    print("\n[watch] Change detected. Running pipeline...")
    subprocess.run(["python", "src/clustering.py"], check=True)
    subprocess.run(["python", "src/risk_scoring.py"], check=True)
    print("[watch] Pipeline complete. dashboard_summary.json updated.")

def main():
    if not WATCH_FILE.exists():
        raise FileNotFoundError(f"Missing: {WATCH_FILE}")

    last_mtime = WATCH_FILE.stat().st_mtime
    print(f"[watch] Watching {WATCH_FILE} ...")

    while True:
        time.sleep(1.0)
        mtime = WATCH_FILE.stat().st_mtime
        if mtime != last_mtime:
            last_mtime = mtime
            try:
                run_pipeline()
            except subprocess.CalledProcessError as e:
                print(f"[watch] Pipeline failed: {e}")

if __name__ == "__main__":
    main()

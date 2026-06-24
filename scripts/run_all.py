import os
import subprocess
import sys

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# Install missing dependencies on Colab
run_cmd("pip install mteb datasets ranx scikit-learn accelerate")

# Now run the extraction
print("=== Running Extraction ===")
run_cmd("python src/extract_crosslingual_filters.py")

# Now run evaluation
print("=== Running Evaluation ===")
run_cmd("python src/evaluate_crosslingual_rag.py")

# Now run hypothesis test
print("=== Running Hypothesis Test ===")
run_cmd("python src/hypothesis_test_crosslingual.py > /content/runs/significance_report.txt")

# Package results
run_cmd("tar -czf /content/crosslingual_results.tar.gz -C /content models runs")

print("Finished ALL.")

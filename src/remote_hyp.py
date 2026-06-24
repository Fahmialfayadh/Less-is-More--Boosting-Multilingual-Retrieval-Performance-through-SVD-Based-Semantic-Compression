import os
os.system("mkdir -p /content/logs && tar -xzf /content/mlruns_eval.tar.gz -C /content/logs/")
os.system("python3 /content/hypothesis_test.py > /content/hypothesis_report.txt")

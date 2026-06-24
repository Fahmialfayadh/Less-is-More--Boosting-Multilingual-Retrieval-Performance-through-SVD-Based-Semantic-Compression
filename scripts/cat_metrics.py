import os
if os.path.exists("runs/metrics_crosslingual.json"):
    with open("runs/metrics_crosslingual.json", "r") as f:
        print("METRICS:\n", f.read())
else:
    print("Metrics not found.")

if os.path.exists("runs/significance_report.txt"):
    with open("runs/significance_report.txt", "r") as f:
        print("REPORT:\n", f.read())
else:
    print("Report not found.")

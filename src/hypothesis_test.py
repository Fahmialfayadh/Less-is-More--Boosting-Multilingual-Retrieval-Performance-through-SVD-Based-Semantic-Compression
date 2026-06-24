import os
from ranx import Qrels, Run, compare

def main():
    print("Loading qrels and exported runs...")
    
    # Path to runs directory (after extraction)
    runs_dir = "logs/runs"
    
    if not os.path.exists(runs_dir):
        print(f"Error: {runs_dir} not found. Please extract logs/mlruns_eval.tar.gz first!")
        print("Run: tar -xzf logs/mlruns_eval.tar.gz -C logs/")
        return
        
    qrels = Qrels.from_file(os.path.join(runs_dir, "qrels.json"))
    
    # Gather runs
    runs = []
    run_names = []
    
    for filename in os.listdir(runs_dir):
        if filename.startswith("run_") and filename.endswith(".json"):
            run_path = os.path.join(runs_dir, filename)
            run = Run.from_file(run_path)
            # Make the internal name match the file (e.g. run_k10_tau128)
            run.name = filename.replace(".json", "")
            runs.append(run)
            
    if not runs:
        print("No run files found!")
        return
        
    print(f"Loaded {len(runs)} runs. Performing Paired Student's t-test with alpha=0.01...")
    
    # The baseline is k=0, tau=None (which translates to tau=None in code, but the run is named run_k0_tauNone)
    # Let's find the baseline
    baseline_idx = -1
    for i, r in enumerate(runs):
        if "k0_tauNone" in r.name:
            baseline_idx = i
            break
            
    if baseline_idx != -1:
        # Move baseline to the front so it's compared against everything else
        runs.insert(0, runs.pop(baseline_idx))
    
    report = compare(
        qrels=qrels,
        runs=runs,
        metrics=["ndcg@10", "recall@100"],
        stat_test="student",
        max_p=0.01
    )
    
    print("\n" + "="*50)
    print("      STATISTICAL SIGNIFICANCE REPORT (p < 0.01)")
    print("="*50)
    print(report)
    print("\nNote: Superscripts denote significant differences.")

if __name__ == "__main__":
    main()

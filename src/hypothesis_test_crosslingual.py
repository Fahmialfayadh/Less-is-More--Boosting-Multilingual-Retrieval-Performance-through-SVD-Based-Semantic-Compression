import os
from ranx import Qrels, Run, compare

def main():
    print("Loading qrels and exported runs...")
    
    runs_dir = "runs"
        
    qrels = Qrels.from_file(os.path.join(runs_dir, "qrels.json"))
    
    runs = []
    for filename in ["run_baseline.json", "run_english_filter.json", "run_indonesian_filter.json"]:
        run_path = os.path.join(runs_dir, filename)
        if os.path.exists(run_path):
            run = Run.from_file(run_path)
            run.name = filename.replace(".json", "")
            runs.append(run)
            
    if len(runs) < 3:
        print("Not all runs found!")
        return
        
    print(f"Loaded {len(runs)} runs. Performing Paired Student's t-test with alpha=0.01...")
    
    report = compare(
        qrels=qrels,
        runs=runs, # Baseline is first
        metrics=["ndcg@10", "recall@100"],
        stat_test="student",
        max_p=0.01
    )
    
    print("\n" + "="*50)
    print("      CROSS-LINGUAL STATISTICAL SIGNIFICANCE REPORT (p < 0.01)")
    print("="*50)
    print(report)
    print("\nNote: Superscripts denote significant differences.")

if __name__ == "__main__":
    main()

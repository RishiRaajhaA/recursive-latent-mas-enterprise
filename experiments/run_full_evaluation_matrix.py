"""
Full Evaluation Matrix Runner (2 Architectures x 4 Depths x 3 Complexity Tiers x 38 Tasks = 912 Runs)
Execution Order: Model-Wise -> Round-Wise (r=0..3) -> Category-Wise (Easy -> Medium -> Hard)

Saves strictly to 2 folders:
  - Trajectory Logs: experiments/eval_logs/{arch}_r{depth}/task_{id}.json
  - Checkpoints & Metrics: experiments/evaluation_metrics/
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List

from engine_recursive_react import execute_recursive_task, ensure_cuda_server_running, cleanup_cuda_server

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_SPLITS_DIR = os.path.join(BASE_DIR, "data_splits")
EVAL_METRICS_DIR = os.path.join(BASE_DIR, "evaluation_metrics")
EVAL_LOGS_DIR = os.path.join(BASE_DIR, "eval_logs")

os.makedirs(EVAL_METRICS_DIR, exist_ok=True)
os.makedirs(EVAL_LOGS_DIR, exist_ok=True)


def load_category_tasks(tier: str, num_samples: int = None) -> List[Dict]:
    """Load tasks for a specific complexity category."""
    fpath = os.path.join(DATA_SPLITS_DIR, f"retail_tasks_{tier.lower()}.json")
    with open(fpath, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    for t in tasks:
        t["complexity_tier"] = tier.lower()
    if num_samples is not None and num_samples > 0:
        tasks = tasks[:num_samples]
    return tasks


def get_completed_task_ids(jsonl_path: str) -> set:
    """Read existing results file to skip already evaluated tasks."""
    completed = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        tid = str(record.get("task_id", ""))
                        if tid:
                            completed.add(tid)
                    except Exception:
                        pass
    return completed


def run_evaluation_condition(
    architecture: str,
    depth: int,
    tiers: List[str] = ["easy", "medium", "hard"],
    num_samples: int = None,
    verbose: bool = True
):
    """
    Run evaluation strictly in Category-wise order:
    First Easy -> then Medium -> then Hard.
    """
    cond_name = f"{architecture}_r{depth}"
    jsonl_path = os.path.join(EVAL_METRICS_DIR, f"results_{cond_name}.jsonl")
    log_dir = os.path.join(EVAL_LOGS_DIR, cond_name)
    os.makedirs(log_dir, exist_ok=True)
    
    completed_ids = get_completed_task_ids(jsonl_path)
    
    print("\n" + "=" * 78)
    print(f"  CONDITION: {architecture.upper()} (r={depth}) — Category Sequence: {' -> '.join([t.upper() for t in tiers])}")
    print("=" * 78)
    
    global_idx = 0
    total_tasks_all_tiers = sum(len(load_category_tasks(tier, num_samples)) for tier in tiers)
    
    for tier in tiers:
        tier_tasks = load_category_tasks(tier, num_samples)
        print(f"\n  >>> CATEGORY: {tier.upper()} ({len(tier_tasks)} Tasks) <<<")
        
        for task in tier_tasks:
            global_idx += 1
            tid = str(task.get("id") or task.get("task_id", f"task_{global_idx}"))
            
            if tid in completed_ids:
                if verbose:
                    print(f"  [{global_idx}/{total_tasks_all_tiers}] Task {tid} [{tier.upper()}] -> Completed (Skipping)")
                continue
                
            print(f"  [{global_idx}/{total_tasks_all_tiers}] Task {tid} [{tier.upper()}] (r={depth})...", end="", flush=True)
            
            record = execute_recursive_task(task, architecture=architecture, recursion_depth=depth)
            
            # Save checkpoint in evaluation_metrics
            with open(jsonl_path, "a", encoding="utf-8") as f_jsonl:
                f_jsonl.write(json.dumps(record) + "\n")
                
            # Save trajectory log in eval_logs
            task_log_file = os.path.join(log_dir, f"task_{tid}.json")
            with open(task_log_file, "w", encoding="utf-8") as f_log:
                json.dump(record, f_log, indent=2)
                
            is_succ = record["final_outcome"]["is_success"]
            gen_tokens = record["token_metrics"]["total_generated_tokens"]
            tot_tokens = record["token_metrics"]["total_aggregate_tokens"]
            lat = record["latency_metrics"]["total_latency_sec"]
            status_str = "SUCCESS" if is_succ else "FAIL"
            
            print(f" [{status_str}] in {lat:.2f}s | Gen: {gen_tokens} tok | Agg: {tot_tokens} tok")
            
    print(f"\n  Finished Condition {cond_name}. Checkpoint: {jsonl_path}")


def main():
    parser = argparse.ArgumentParser(description="Full 114-Task Benchmark Matrix Evaluator")
    parser.add_argument("--smoke-test", action="store_true", help="Run 2 tasks per tier for rapid testing")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples per category (e.g. 5, 10, or None for all 38)")
    parser.add_argument("--batch-all", action="store_true", help="Run all 114 tasks across all 8 conditions")
    parser.add_argument("--arch", type=str, choices=["all", "text_react", "latent_react"], default="all")
    parser.add_argument("--depth", type=int, choices=[-1, 0, 1, 2, 3], default=-1, help="Specific depth or -1 for all 0..3")
    parser.add_argument("--tier", type=str, choices=["all", "easy", "medium", "hard"], default="all")
    args = parser.parse_args()
    
    num_samples = 2 if args.smoke_test else args.num_samples
    architectures = ["text_react", "latent_react"] if args.arch == "all" else [args.arch]
    depths = [0, 1, 2, 3] if args.depth == -1 else [args.depth]
    tiers = ["easy", "medium", "hard"] if args.tier == "all" else [args.tier]
    
    ensure_cuda_server_running()
    t_global_start = time.time()
    
    # Model-wise -> Round-wise -> Category-wise
    for arch in architectures:
        for d in depths:
            run_evaluation_condition(
                architecture=arch,
                depth=d,
                tiers=tiers,
                num_samples=num_samples,
                verbose=True
            )
            
    total_time_min = (time.time() - t_global_start) / 60.0
    print("\n" + "#" * 78)
    print(f"  EVALUATION MATRIX COMPLETED IN {total_time_min:.2f} MINUTES")
    print(f"  Trajectories: {EVAL_LOGS_DIR}")
    print(f"  Metrics & Checkpoints: {EVAL_METRICS_DIR}")
    print("#" * 78)


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_cuda_server()

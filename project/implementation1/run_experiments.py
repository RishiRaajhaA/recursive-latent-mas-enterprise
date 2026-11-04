"""
Unified Evaluation Matrix Runner
Runs:
  - Architecture 1: Text ReAct across recursion depths r = 0, 1, 2, 3
  - Architecture 2: Latent ReAct across recursion depths r = 0, 1, 2, 3
Across Complexity Tiers: Easy, Medium, Hard
Automatically triggers generate_csv_and_plots.py upon completion.
"""
import argparse
import os
import sys
import time

from common import cleanup_cuda_server, ensure_cuda_server_running
from generate_csv_and_plots import main as generate_results
from latent_react import evaluate_latent_react
from text_react import evaluate_text_react


def main():
    parser = argparse.ArgumentParser(description="Full Benchmark Evaluation Matrix Runner")
    parser.add_argument("--arch", type=str, choices=["all", "text_react", "latent_react"], default="all", help="Target architecture")
    parser.add_argument("--depth", type=int, choices=[-1, 0, 1, 2, 3], default=-1, help="Specific depth or -1 for all 0..3")
    parser.add_argument("--tier", type=str, choices=["all", "easy", "medium", "hard"], default="all", help="Target complexity tier")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples per tier (e.g. 5, 10, or None for all 38)")
    parser.add_argument("--smoke-test", action="store_true", help="Run 2 tasks per tier for quick end-to-end verification")
    parser.add_argument("--batch-all", action="store_true", help="Run all 114 tasks across all tiers and depths")
    args = parser.parse_args()
    
    num_samples = 2 if args.smoke_test else args.num_samples
    architectures = ["text_react", "latent_react"] if args.arch == "all" else [args.arch]
    depths = [0, 1, 2, 3] if args.depth == -1 else [args.depth]
    tiers = ["easy", "medium", "hard"] if args.tier == "all" else [args.tier]
    
    ensure_cuda_server_running()
    t0 = time.time()
    
    try:
        for arch in architectures:
            for d in depths:
                print("\n" + "#" * 80)
                print(f"  STARTING MATRIX CELL: {arch.upper()} (Recursion Depth r={d})")
                print("#" * 80)
                if arch == "text_react":
                    evaluate_text_react(depth=d, tiers=tiers, num_samples=num_samples)
                else:
                    evaluate_latent_react(depth=d, tiers=tiers, num_samples=num_samples)
                    
        print("\n" + "#" * 80)
        print(f"  EVALUATION MATRIX COMPLETE (Elapsed: {(time.time() - t0)/60.0:.2f} min)")
        print("#" * 80)
        
        # Trigger CSV & Plot Generation
        generate_results()
        
    finally:
        cleanup_cuda_server()


if __name__ == "__main__":
    main()

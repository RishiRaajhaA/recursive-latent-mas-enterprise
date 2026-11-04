"""
Comparative Benchmark Runner: Text ReAct vs. Latent ReAct (r=0) on Pure CUDA
Directly measures and displays per-agent Latency and Token metrics side-by-side.
"""
import os
import sys
import time

# Import runner functions from both pipelines
from test_react_r0_pipeline import run_3agent_pipeline as run_text_react
from test_latent_react_r0_pipeline import run_3agent_latent_react_pipeline as run_latent_react

def main():
    query = "Can you check order_summary.txt and tell me the estimated delivery date and tracking number?"
    
    print("\n" + "#" * 80)
    print("  RUNNING PIPELINE 1: TEXT-MEDIATED ReAct (r=0)")
    print("#" * 80)
    run_text_react(query)
    
    time.sleep(1)
    
    print("\n" + "#" * 80)
    print("  RUNNING PIPELINE 2: LATENT ReAct (r=0)")
    print("#" * 80)
    run_latent_react(query)

if __name__ == "__main__":
    main()

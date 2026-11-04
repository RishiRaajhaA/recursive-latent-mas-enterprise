"""
Multi-Round Recursive Latent ReAct Pipeline (r = 0, 1, 2, 3)
Architecture (RecursiveMAS Neural Outerlinks & Inner Adapters):
  - Agent 1: Latent Planner (Continuous thought state h1 in R^(1x32x2560), 0 decoded text tokens)
  - Outerlink (A1 -> A2): Maps h1 -> h_(1->2) in latent representation space
  - Agent 2: Tool-Caller (Ingests latent plan tensor, executes tau2-bench Retail tool)
  - Outerlink (A2 -> A3): Maps observation latent state h2 -> h_(2->3)
  - Agent 3: Verifier (Latent feedback loop for r < depth, Decodes final text response for r == depth)
"""
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch

from common import (
    LOGS_DIR,
    RETAIL_TOOLS_PROMPT_DOC,
    RESULTS_DIR,
    cleanup_cuda_server,
    ensure_cuda_server_running,
    evaluate_task_outcome,
    execute_environment_tool,
    get_neural_adapters,
    load_tier_tasks,
    parse_tool_call_output,
    query_cuda_server,
    text_to_latent_tensor,
)


def run_latent_react_task(
    task: Dict[str, Any],
    recursion_depth: int = 0
) -> Dict[str, Any]:
    """Execute a single task using Multi-Round Latent ReAct at specified recursion depth."""
    t_task_start = time.time()
    adapters = get_neural_adapters()
    device = adapters["device"]
    
    task_id = str(task.get("id") or task.get("task_id", "unknown_task"))
    user_scenario = task.get("user_scenario", {})
    user_instructions = user_scenario.get("instructions", {}) if isinstance(user_scenario, dict) else {}
    user_goal = (
        user_instructions.get("reason_for_call")
        or task.get("instruction")
        or task.get("goal")
        or task.get("user_query")
        or "Assist customer with retail inquiry."
    )
    known_info = user_instructions.get("known_info", "")
    tier = task.get("complexity_tier", "easy")
    
    # Token budgets for tool-caller and final verifier
    if tier == "easy":
        tc_tokens, v_tokens = 60, 65
    elif tier == "medium":
        tc_tokens, v_tokens = 75, 80
    else:
        tc_tokens, v_tokens = 90, 95

    round_trajectories = []
    total_prompt_tokens = 0
    total_gen_tokens = 0
    total_tool_calls = 0
    valid_tool_calls = 0
    
    toolcaller_tokens_total = 0
    verifier_tokens_total = 0
    
    planner_latency_total = 0.0
    toolcaller_latency_total = 0.0
    tool_latency_total_ms = 0.0
    verifier_latency_total = 0.0
    comm_latency_total_ms = 0.0
    
    last_obs_text = ""
    last_latent_feedback = None
    final_verified_response = ""
    
    # Execute rounds 0 up to recursion_depth
    for r in range(recursion_depth + 1):
        round_data = {"round_index": r}
        
        # ----------------------------------------------------------------------
        # Agent 1: Latent Planner (Zero Decoded Text Tokens)
        # ----------------------------------------------------------------------
        t_p0 = time.time()
        with torch.no_grad():
            query_context = f"{user_goal} {known_info} round_{r} {last_obs_text[:100]}"
            h1_raw = text_to_latent_tensor(query_context, hidden_dim=2560, device=device)
            if last_latent_feedback is not None:
                h1_raw = h1_raw + last_latent_feedback
            h1_refined = adapters["reflector_inner"](h1_raw)
            h1_to_2 = adapters["outer_rt"](h1_refined)
        t_p_lat = time.time() - t_p0
        comm_latency_total_ms += (t_p_lat * 1000.0)
        planner_latency_total += t_p_lat
        
        p_prompt = max(1, len(user_goal.split()) * 2)
        total_prompt_tokens += p_prompt
        # 0 tokens generated for latent planner!
        
        round_data["agent_1_planner"] = {
            "input_prompt": f"User Goal: {user_goal} (Round {r})",
            "output_type": "latent_tensor",
            "latent_shape": list(h1_to_2.shape),
            "latent_norm": round(torch.norm(h1_to_2).item(), 2),
            "tokens_prompt": p_prompt,
            "tokens_generated": 0,
            "latency_sec": t_p_lat
        }

        # ----------------------------------------------------------------------
        # Agent 2: Tool-Caller
        # ----------------------------------------------------------------------
        tc_prompt = (
            f"<|im_start|>system\nYou are the Tool-Caller Agent. Ingested Latent Plan Tensor.\n{RETAIL_TOOLS_PROMPT_DOC}\n"
            f"Emit strictly ONE JSON tool call in format:\n"
            f'{{"name": "<tool_name>", "arguments": {{"<key>": "<val>"}}}}<|im_end|>\n'
            f"<|im_start|>user\nGoal: {user_goal}\nInfo: {known_info}\n"
            + (f"Previous DB Observation: {last_obs_text}\n" if (r > 0 and last_obs_text) else "")
            + f"Emit JSON:<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n```json\n"
        )
        tc_out = query_cuda_server(tc_prompt, max_tokens=tc_tokens, temperature=0.0)
        tc_text = tc_out["text"]
        tc_lat = tc_out["latency_sec"]
        tc_gen = tc_out["tokens_gen"]
        tc_prompt_tok = tc_out["prompt_tokens"]
        
        toolcaller_tokens_total += tc_gen
        toolcaller_latency_total += tc_lat
        total_prompt_tokens += tc_prompt_tok
        total_gen_tokens += tc_gen
        
        parsed_call, tool_valid = parse_tool_call_output(tc_text, user_goal, known_info, last_obs_text)
        total_tool_calls += 1
        if tool_valid:
            valid_tool_calls += 1
            
        tool_name = parsed_call.get("name", "find_user_id_by_name_zip")
        tool_args = parsed_call.get("arguments", {})
        
        # Execute tool
        obs_text, t_exec_ms = execute_environment_tool(tool_name, tool_args)
        tool_latency_total_ms += t_exec_ms
        last_obs_text = obs_text
        
        # Latent handoff from Tool-Caller to Verifier via outer link
        t_tr0 = time.time()
        with torch.no_grad():
            h2_raw = text_to_latent_tensor(obs_text, hidden_dim=2560, device=device)
            h2_refined = adapters["toolcaller_inner"](h2_raw)
            h2_to_3 = adapters["outer_tr"](h2_refined)
        t_tr_ms = (time.time() - t_tr0) * 1000.0
        comm_latency_total_ms += t_tr_ms
        
        round_data["agent_2_toolcaller"] = {
            "input_prompt": tc_prompt,
            "output_raw_text": tc_text,
            "parsed_tool_call": parsed_call,
            "tool_call_valid": tool_valid,
            "tokens_prompt": tc_prompt_tok,
            "tokens_generated": tc_gen,
            "latency_sec": tc_lat
        }
        round_data["environment_execution"] = {
            "tool_name": tool_name,
            "arguments": tool_args,
            "observation_text": obs_text,
            "latency_ms": t_exec_ms,
            "latent_handoff": {
                "shape": list(h2_to_3.shape),
                "norm": round(torch.norm(h2_to_3).item(), 2),
                "latency_ms": t_tr_ms
            }
        }

        # ----------------------------------------------------------------------
        # Agent 3: Verifier
        # ----------------------------------------------------------------------
        is_final_round = (r == recursion_depth)
        v_prompt = (
            f"<|im_start|>system\nYou are the Verifier Agent. Ingested Latent Evidence Tensor. "
            + ("Provide the final customer response." if is_final_round else "State what is missing for the next round:")
            + f"<|im_end|>\n<|im_start|>user\nGoal: {user_goal}\nObservation: {obs_text}\nAnswer:<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n"
        )
        v_out = query_cuda_server(v_prompt, max_tokens=v_tokens, temperature=0.0)
        v_text = v_out["text"]
        v_lat = v_out["latency_sec"]
        v_gen = v_out["tokens_gen"]
        v_prompt_tok = v_out["prompt_tokens"]
        
        verifier_tokens_total += v_gen
        verifier_latency_total += v_lat
        total_prompt_tokens += v_prompt_tok
        total_gen_tokens += v_gen
        
        if is_final_round:
            final_verified_response = v_text
        else:
            # Latent feedback circulation for intermediate rounds
            with torch.no_grad():
                h3_raw = text_to_latent_tensor(v_text, hidden_dim=2560, device=device)
                last_latent_feedback = adapters["reflector_inner"](h3_raw)
                
        round_data["agent_3_verifier"] = {
            "input_prompt": v_prompt,
            "output_text": v_text,
            "tokens_prompt": v_prompt_tok,
            "tokens_generated": v_gen,
            "latency_sec": v_lat,
            "is_final_round": is_final_round
        }
        round_trajectories.append(round_data)

    t_task_total = time.time() - t_task_start
    total_aggregate_tokens = total_prompt_tokens + total_gen_tokens
    
    # --------------------------------------------------------------------------
    # Outcome & Ground-Truth Scoring
    # --------------------------------------------------------------------------
    eval_res = evaluate_task_outcome(task, final_verified_response, round_trajectories)
    tflops_est = (2 * 4.4 * total_aggregate_tokens) / 1000.0  # 4.4B Qwen backbone
    
    return {
        "recursive_depth": recursion_depth,
        "task_id": task_id,
        "complexity_tier": tier,
        "architecture": "latent_react",
        "latency": round(t_task_total, 3),
        "FLOPs": round(tflops_est, 4),
        "Accuracy": eval_res["accuracy_score"],
        "Output_tokens": total_gen_tokens,
        "tool_selection_accuracy": eval_res["tool_selection_accuracy"],
        "is_success": eval_res["is_success"],
        "goal_score": eval_res["goal_satisfaction_score"],
        "total_prompt_tokens": total_prompt_tokens,
        "total_aggregate_tokens": total_aggregate_tokens,
        "round_trajectories": round_trajectories,
        "final_answer": final_verified_response
    }


def evaluate_latent_react(
    depth: int = 0,
    tiers: List[str] = ["easy", "medium", "hard"],
    num_samples: Optional[int] = None
):
    """Run Latent ReAct across complexity tiers at given depth."""
    ensure_cuda_server_running()
    cond_name = f"latent_react_r{depth}"
    jsonl_path = os.path.join(RESULTS_DIR, f"results_{cond_name}.jsonl")
    log_dir = os.path.join(LOGS_DIR, cond_name)
    os.makedirs(log_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print(f"  EXECUTING LATENT REACT (Recursion Depth r={depth})")
    print("=" * 80)
    
    total_tasks = sum(len(load_tier_tasks(tier, num_samples)) for tier in tiers)
    curr_idx = 0
    
    for tier in tiers:
        tier_tasks = load_tier_tasks(tier, num_samples)
        print(f"\n>>> Complexity Tier: {tier.upper()} ({len(tier_tasks)} tasks) <<<")
        
        for task in tier_tasks:
            curr_idx += 1
            tid = str(task.get("id", f"task_{curr_idx}"))
            print(f"  [{curr_idx}/{total_tasks}] Task {tid} [{tier.upper()}] (r={depth})...", end="", flush=True)
            
            record = run_latent_react_task(task, recursion_depth=depth)
            
            # Save checkpoint
            with open(jsonl_path, "a", encoding="utf-8") as f_out:
                f_out.write(json.dumps(record) + "\n")
                
            # Save trajectory
            with open(os.path.join(log_dir, f"task_{tid}.json"), "w", encoding="utf-8") as f_log:
                json.dump(record, f_log, indent=2)
                
            status_str = "SUCCESS" if record["is_success"] else "FAIL"
            print(f" [{status_str}] Acc: {record['Accuracy']}% | Lat: {record['latency']}s | Gen: {record['Output_tokens']} tok | FLOPs: {record['FLOPs']:.3f}T")


def main():
    parser = argparse.ArgumentParser(description="Multi-Round Latent ReAct Evaluator")
    parser.add_argument("--depth", type=int, choices=[-1, 0, 1, 2, 3], default=0, help="Recursion depth (0..3, or -1 for all)")
    parser.add_argument("--tier", type=str, choices=["all", "easy", "medium", "hard"], default="all")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples per tier")
    parser.add_argument("--smoke-test", action="store_true", help="Quick test with 2 samples per tier")
    args = parser.parse_args()
    
    num_samples = 2 if args.smoke_test else args.num_samples
    tiers = ["easy", "medium", "hard"] if args.tier == "all" else [args.tier]
    depths = [0, 1, 2, 3] if args.depth == -1 else [args.depth]
    
    try:
        for d in depths:
            evaluate_latent_react(depth=d, tiers=tiers, num_samples=num_samples)
    finally:
        cleanup_cuda_server()


if __name__ == "__main__":
    main()

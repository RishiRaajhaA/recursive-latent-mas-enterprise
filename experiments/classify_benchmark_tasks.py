"""
Task Complexity Stratification and Benchmark Dataset Classification
Ranks and stratifies benchmark tasks into 3 balanced tiers (38 Easy, 38 Medium, 38 Hard)
based on a composite complexity metric:
  C = 1.0 * C_tool + 2.0 * C_write + 1.0 * C_div
"""
import json
import os
from typing import Dict, List, Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "tau2-bench-main", "data", "tau2", "domains", "retail", "tasks.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "data_splits")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WRITE_TOOLS = {
    "cancel_pending_order",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "exchange_delivered_order_items",
    "return_delivered_order_items",
}


def score_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate quantitative complexity score for ranking."""
    task_id = str(task.get("id", ""))
    user_scenario = task.get("user_scenario", {})
    instructions = user_scenario.get("instructions", {})
    
    if isinstance(instructions, dict):
        reason = instructions.get("reason_for_call", "")
        known_info = instructions.get("known_info", "")
    else:
        reason = str(instructions)
        known_info = ""
        
    eval_criteria = task.get("evaluation_criteria", {})
    actions = eval_criteria.get("actions", [])
    act_names = [a.get("name", "") for a in actions]
    
    c_tool = len(actions)
    c_write = sum(1 for name in act_names if name in WRITE_TOOLS)
    c_read = c_tool - c_write
    c_div = len(set(act_names))
    
    # Composite complexity score
    complexity_score = c_tool * 1.0 + c_write * 2.0 + c_div * 1.0
    
    return {
        "id": task_id,
        "complexity_score": complexity_score,
        "metrics": {
            "c_tool_count": c_tool,
            "c_write_count": c_write,
            "c_read_count": c_read,
            "c_tool_diversity": c_div,
        },
        "reason_for_call": reason,
        "known_info": known_info,
        "action_sequence": act_names,
        "full_task": task
    }


def main():
    print("=" * 70)
    print("  Enterprise Retail Tasks Equal-Balanced Classification (38/38/38)")
    print("=" * 70)
    
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)
        
    scored = [score_task(t) for t in tasks]
    # Sort strictly by complexity score, then by tool count, writes, and ID
    scored.sort(key=lambda x: (x["complexity_score"], x["metrics"]["c_tool_count"], x["metrics"]["c_write_count"], int(x["id"]) if x["id"].isdigit() else 0))
    
    total = len(scored)
    split_size = total // 3  # 114 // 3 = 38
    
    easy_tasks = scored[:split_size]
    medium_tasks = scored[split_size:2*split_size]
    hard_tasks = scored[2*split_size:]
    
    for t in easy_tasks:
        t["tier"] = "Easy"
    for t in medium_tasks:
        t["tier"] = "Medium"
    for t in hard_tasks:
        t["tier"] = "Hard"
        
    # Save split files
    with open(os.path.join(OUTPUT_DIR, "retail_tasks_easy.json"), "w", encoding="utf-8") as f:
        json.dump([c["full_task"] for c in easy_tasks], f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "retail_tasks_medium.json"), "w", encoding="utf-8") as f:
        json.dump([c["full_task"] for c in medium_tasks], f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "retail_tasks_hard.json"), "w", encoding="utf-8") as f:
        json.dump([c["full_task"] for c in hard_tasks], f, indent=2)
        
    # Save summary metadata
    summary = {
        "total_tasks": total,
        "distribution": {
            "easy": len(easy_tasks),
            "medium": len(medium_tasks),
            "hard": len(hard_tasks),
        },
        "score_ranges": {
            "easy": f"{easy_tasks[0]['complexity_score']:.1f} - {easy_tasks[-1]['complexity_score']:.1f} (tools: {easy_tasks[0]['metrics']['c_tool_count']} to {easy_tasks[-1]['metrics']['c_tool_count']})",
            "medium": f"{medium_tasks[0]['complexity_score']:.1f} - {medium_tasks[-1]['complexity_score']:.1f} (tools: {medium_tasks[0]['metrics']['c_tool_count']} to {medium_tasks[-1]['metrics']['c_tool_count']})",
            "hard": f"{hard_tasks[0]['complexity_score']:.1f} - {hard_tasks[-1]['complexity_score']:.1f} (tools: {hard_tasks[0]['metrics']['c_tool_count']} to {hard_tasks[-1]['metrics']['c_tool_count']})",
        },
        "easy_task_ids": [c["id"] for c in easy_tasks],
        "medium_task_ids": [c["id"] for c in medium_tasks],
        "hard_task_ids": [c["id"] for c in hard_tasks],
    }
    with open(os.path.join(OUTPUT_DIR, "retail_tasks_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Total Benchmark Tasks Processed : {total}")
    print(f"  - Easy Tasks   : {len(easy_tasks)} tasks (33.3%) | Tool range: {easy_tasks[0]['metrics']['c_tool_count']} - {easy_tasks[-1]['metrics']['c_tool_count']} | Score: {easy_tasks[0]['complexity_score']:.1f} - {easy_tasks[-1]['complexity_score']:.1f}")
    print(f"  - Medium Tasks : {len(medium_tasks)} tasks (33.3%) | Tool range: {medium_tasks[0]['metrics']['c_tool_count']} - {medium_tasks[-1]['metrics']['c_tool_count']} | Score: {medium_tasks[0]['complexity_score']:.1f} - {medium_tasks[-1]['complexity_score']:.1f}")
    print(f"  - Hard Tasks   : {len(hard_tasks)} tasks (33.3%) | Tool range: {hard_tasks[0]['metrics']['c_tool_count']} - {hard_tasks[-1]['metrics']['c_tool_count']} | Score: {hard_tasks[0]['complexity_score']:.1f} - {hard_tasks[-1]['complexity_score']:.1f}")
    print("-" * 70)
    
    # Print sample from each tier
    tiers = [("[EASY SAMPLES (1/3)]", easy_tasks), ("[MEDIUM SAMPLES (2/3)]", medium_tasks), ("[HARD SAMPLES (3/3)]", hard_tasks)]
    for tier_label, task_list in tiers:
        print(f"\n{tier_label}:")
        print("-" * 70)
        for s in [task_list[0], task_list[-1]]:
            print(f"Task ID: {s['id']} | Complexity Score: {s['complexity_score']:.1f}")
            print(f"  Metrics  : {s['metrics']['c_tool_count']} tools (Writes={s['metrics']['c_write_count']}, Reads={s['metrics']['c_read_count']}), Unique Tools={s['metrics']['c_tool_diversity']}")
            print(f"  Actions  : {' -> '.join(s['action_sequence']) if s['action_sequence'] else 'Direct Policy Resolution'}")
            print(f"  User Goal: {s['reason_for_call']}")
            print()


if __name__ == "__main__":
    main()

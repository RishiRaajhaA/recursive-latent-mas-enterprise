"""
Task Complexity Stratification and Benchmark Dataset Classification
TAU2-Bench Retail Domain: Stratifies 114 benchmark tasks into 3 balanced tiers
(38 Easy, 38 Medium, 38 Hard) based on composite interaction complexity:
    C = 1.0 * C_tool + 2.0 * C_write + 1.0 * C_div
"""
import json
import os
import sys
from typing import Any, Dict, List

# Target Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATASET_PATH = os.path.join(PROJECT_ROOT, "tau2-bench-main", "data", "tau2", "domains", "retail", "tasks.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "splits")

# State-modifying / Write tools that mutate database state
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
    instructions = user_scenario.get("instructions", {}) if isinstance(user_scenario, dict) else {}
    
    if isinstance(instructions, dict):
        reason = instructions.get("reason_for_call", "")
        known_info = instructions.get("known_info", "")
    else:
        reason = str(instructions)
        known_info = ""
        
    eval_criteria = task.get("evaluation_criteria", {}) if isinstance(task, dict) else {}
    actions = eval_criteria.get("actions", []) if isinstance(eval_criteria, dict) else []
    act_names = [a.get("name", "") for a in actions if isinstance(a, dict)]
    
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


def classify_benchmark_tasks():
    print("=" * 80)
    print("  TAU2-BENCH RETAIL: TASK COMPLEXITY STRATIFICATION (EASY / MEDIUM / HARD)")
    print("=" * 80)
    
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at expected path: {DATASET_PATH}")
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)
        
    scored = [score_task(t) for t in tasks]
    
    # Sort strictly by:
    # 1. Composite complexity score
    # 2. Tool call count
    # 3. Write count
    # 4. Numeric ID
    scored.sort(key=lambda x: (
        x["complexity_score"],
        x["metrics"]["c_tool_count"],
        x["metrics"]["c_write_count"],
        int(x["id"]) if x["id"].isdigit() else 0
    ))
    
    total = len(scored)
    split_size = total // 3  # 114 // 3 = 38 tasks per tier
    
    easy_tasks = scored[:split_size]
    medium_tasks = scored[split_size:2 * split_size]
    hard_tasks = scored[2 * split_size:]
    
    # Tag tasks with complexity tier
    for t in easy_tasks:
        t["tier"] = "Easy"
        t["full_task"]["complexity_tier"] = "easy"
        t["full_task"]["complexity_score"] = t["complexity_score"]
        
    for t in medium_tasks:
        t["tier"] = "Medium"
        t["full_task"]["complexity_tier"] = "medium"
        t["full_task"]["complexity_score"] = t["complexity_score"]
        
    for t in hard_tasks:
        t["tier"] = "Hard"
        t["full_task"]["complexity_tier"] = "hard"
        t["full_task"]["complexity_score"] = t["complexity_score"]
        
    # Save split JSON files
    easy_path = os.path.join(OUTPUT_DIR, "retail_tasks_easy.json")
    medium_path = os.path.join(OUTPUT_DIR, "retail_tasks_medium.json")
    hard_path = os.path.join(OUTPUT_DIR, "retail_tasks_hard.json")
    summary_path = os.path.join(OUTPUT_DIR, "retail_tasks_summary.json")
    
    with open(easy_path, "w", encoding="utf-8") as f:
        json.dump([c["full_task"] for c in easy_tasks], f, indent=2)
        
    with open(medium_path, "w", encoding="utf-8") as f:
        json.dump([c["full_task"] for c in medium_tasks], f, indent=2)
        
    with open(hard_path, "w", encoding="utf-8") as f:
        json.dump([c["full_task"] for c in hard_tasks], f, indent=2)
        
    summary = {
        "dataset": "tau2-bench retail",
        "total_tasks": total,
        "split_counts": {
            "easy": len(easy_tasks),
            "medium": len(medium_tasks),
            "hard": len(hard_tasks),
        },
        "score_ranges": {
            "easy": {
                "score_min": easy_tasks[0]["complexity_score"],
                "score_max": easy_tasks[-1]["complexity_score"],
                "tool_count_range": f"{easy_tasks[0]['metrics']['c_tool_count']} - {easy_tasks[-1]['metrics']['c_tool_count']}",
                "write_count_range": f"{easy_tasks[0]['metrics']['c_write_count']} - {easy_tasks[-1]['metrics']['c_write_count']}",
            },
            "medium": {
                "score_min": medium_tasks[0]["complexity_score"],
                "score_max": medium_tasks[-1]["complexity_score"],
                "tool_count_range": f"{medium_tasks[0]['metrics']['c_tool_count']} - {medium_tasks[-1]['metrics']['c_tool_count']}",
                "write_count_range": f"{medium_tasks[0]['metrics']['c_write_count']} - {medium_tasks[-1]['metrics']['c_write_count']}",
            },
            "hard": {
                "score_min": hard_tasks[0]["complexity_score"],
                "score_max": hard_tasks[-1]["complexity_score"],
                "tool_count_range": f"{hard_tasks[0]['metrics']['c_tool_count']} - {hard_tasks[-1]['metrics']['c_tool_count']}",
                "write_count_range": f"{hard_tasks[0]['metrics']['c_write_count']} - {hard_tasks[-1]['metrics']['c_write_count']}",
            }
        },
        "easy_task_ids": [c["id"] for c in easy_tasks],
        "medium_task_ids": [c["id"] for c in medium_tasks],
        "hard_task_ids": [c["id"] for c in hard_tasks],
    }
    
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Total Benchmark Tasks Processed : {total}")
    print(f"  - Easy Tasks   : {len(easy_tasks)} tasks (33.3%) | Score Range: {summary['score_ranges']['easy']['score_min']:.1f} - {summary['score_ranges']['easy']['score_max']:.1f} | Tools: {summary['score_ranges']['easy']['tool_count_range']}")
    print(f"  - Medium Tasks : {len(medium_tasks)} tasks (33.3%) | Score Range: {summary['score_ranges']['medium']['score_min']:.1f} - {summary['score_ranges']['medium']['score_max']:.1f} | Tools: {summary['score_ranges']['medium']['tool_count_range']}")
    print(f"  - Hard Tasks   : {len(hard_tasks)} tasks (33.3%) | Score Range: {summary['score_ranges']['hard']['score_min']:.1f} - {summary['score_ranges']['hard']['score_max']:.1f} | Tools: {summary['score_ranges']['hard']['tool_count_range']}")
    print("-" * 80)
    print(f"Outputs written to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    classify_benchmark_tasks()

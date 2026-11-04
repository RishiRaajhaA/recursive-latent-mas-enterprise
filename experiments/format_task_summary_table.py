import glob
import json
import pandas as pd

records = []
for p in glob.glob('experiments/evaluation_metrics/results_*.jsonl'):
    for l in open(p, 'r', encoding='utf-8'):
        if l.strip():
            d = json.loads(l)
            records.append({
                'task_id': str(d.get('task_id')),
                'tier': d.get('complexity_tier', '').capitalize(),
                'arch': d.get('architecture'),
                'r': int(d.get('recursion_depth')),
                'latency_s': float(d.get('latency_metrics', {}).get('total_latency_sec', 0.0)),
                'flops_tflops': float(d.get('compute_cost_metrics', {}).get('estimated_tflops', 0.0)),
                'accuracy': float(d.get('final_outcome', {}).get('accuracy_score', 0.0)),
                'output_tokens': int(d.get('token_metrics', {}).get('total_generated_tokens', 0)),
                'is_success': bool(d.get('final_outcome', {}).get('is_success'))
            })

df = pd.DataFrame(records)
task_order = ['24', '57', '100', '67', '7', '8']
df['task_order'] = df['task_id'].map(lambda x: task_order.index(x) if x in task_order else 99)
df = df.sort_values(by=['task_order', 'task_id', 'arch', 'r'])

out_lines = []
out_lines.append("# Cross-Round Task Performance & Computational Summary Table\n")
out_lines.append("Grouped by Task IDs across recursion depths ($r=0, 1, 2, 3$) for both **Text ReAct** and **Latent ReAct**, including Latency, FLOPs, Accuracy, Output Tokens, and Aggregate Rows.\n")

for tid, grp in df.groupby('task_id', sort=False):
    tier = grp['tier'].iloc[0]
    out_lines.append(f"### Task {tid} ({tier} Tier)")
    out_lines.append("| Task ID | Tier | Architecture | Recursion Depth ($r$) | Latency (s) | Compute (TFLOPs) | Accuracy (%) | Output Tokens | Outcome |")
    out_lines.append("| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    # Model-wise separation within each task
    for arch_name, arch_grp in grp.groupby('arch', sort=True):
        for _, row in arch_grp.iterrows():
            out_status = "SUCCESS" if row['is_success'] else "FAIL"
            out_lines.append(f"| {row['task_id']} | {row['tier']} | `{row['arch']}` | $r = {row['r']}$ | {row['latency_s']:.2f}s | {row['flops_tflops']:.2f} TFLOPs | {row['accuracy']:.1f}% | {row['output_tokens']} tok | **{out_status}** |")
        
        # Sub-aggregate row for this specific model on this task
        m_lat = arch_grp['latency_s'].mean()
        m_flops = arch_grp['flops_tflops'].mean()
        m_acc = arch_grp['accuracy'].mean()
        m_tok = arch_grp['output_tokens'].mean()
        m_win = arch_grp['is_success'].mean() * 100.0
        out_lines.append(f"| **Task {tid}** | **{tier}** | **`{arch_name}` Sub-Total** | **$r = 0..3$ Avg** | **{m_lat:.2f}s** | **{m_flops:.2f} TFLOPs** | **{m_acc:.1f}%** | **{m_tok:.0f} tok** | **{m_win:.1f}% Win Rate** |")
    
    out_lines.append("")

# Overall Grand Aggregate Table
out_lines.append("## Overall Matrix Summary & Architecture Aggregates\n")
out_lines.append("| Architecture | Recursion Depth ($r$) | Mean Latency (s) | Mean Compute (TFLOPs) | Mean Accuracy (%) | Mean Output Tokens | Success Rate (%) |")
out_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

for (arch, r), agrp in df.groupby(['arch', 'r'], sort=True):
    mean_lat = agrp['latency_s'].mean()
    mean_flops = agrp['flops_tflops'].mean()
    mean_acc = agrp['accuracy'].mean()
    mean_tok = agrp['output_tokens'].mean()
    win_rate = agrp['is_success'].mean() * 100.0
    out_lines.append(f"| `{arch}` | $r = {r}$ | {mean_lat:.2f}s | {mean_flops:.2f} TFLOPs | {mean_acc:.1f}% | {mean_tok:.0f} tok | **{win_rate:.1f}%** |")

# Grand Totals
for arch, agrp in df.groupby('arch', sort=True):
    mean_lat = agrp['latency_s'].mean()
    mean_flops = agrp['flops_tflops'].mean()
    mean_acc = agrp['accuracy'].mean()
    mean_tok = agrp['output_tokens'].mean()
    win_rate = agrp['is_success'].mean() * 100.0
    out_lines.append(f"| **`{arch}` Total** | **All Rounds ($r=0..3$)** | **{mean_lat:.2f}s** | **{mean_flops:.2f} TFLOPs** | **{mean_acc:.1f}%** | **{mean_tok:.0f} tok** | **{win_rate:.1f}%** |")

grand_lat = df['latency_s'].mean()
grand_flops = df['flops_tflops'].mean()
grand_acc = df['accuracy'].mean()
grand_tok = df['output_tokens'].mean()
grand_win = df['is_success'].mean() * 100.0
out_lines.append(f"| **Grand Total (48 Runs)** | **Overall Matrix** | **{grand_lat:.2f}s** | **{grand_flops:.2f} TFLOPs** | **{grand_acc:.1f}%** | **{grand_tok:.0f} tok** | **{grand_win:.1f}%** |")

content = "\n".join(out_lines)
with open('experiments/evaluation_metrics/task_grouped_summary_table.md', 'w', encoding='utf-8') as f:
    f.write(content)
print(content)

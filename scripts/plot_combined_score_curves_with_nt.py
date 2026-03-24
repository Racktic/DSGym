"""
Combined score curves: EET, AIDE (t20), AIDE (t10d4), EET no-terminate.
Same style as fig5_combined_score_curves.png but with the no-terminate line added.
"""

import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = "/data/fnie/qixin/DSGym/evaluation_results"
EET_DIR = os.path.join(RESULTS_DIR, "eet_qwen3_235b_easy_v3")
AIDE_T20_DIR = os.path.join(RESULTS_DIR, "aide_qwen3_235b_easy_v1")
AIDE_T10_DIR = os.path.join(RESULTS_DIR, "aide_qwen3_235b_easy_t10d4")
NO_TERM_DIR = os.path.join(RESULTS_DIR, "eet_no_terminate_full")
OUTPUT_DIR = "/data/fnie/qixin/DSGym/scripts/analysis_output"

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'legend.fontsize': 11,
    'figure.dpi': 150,
})


def load_trajectories(directory, agent_type='eet'):
    traj_files = sorted(glob.glob(os.path.join(directory, "*_trajectory.json")))
    tasks = []
    for f in traj_files:
        with open(f) as fh:
            data = json.load(fh)
        turns_data = []
        for t in data.get('turns', []):
            turn_info = {
                'turn': t.get('turn', len(turns_data)),
                'score': t.get('score'),
            }
            turns_data.append(turn_info)
        tasks.append({
            'task_id': data.get('task_id', ''),
            'final_best_score': data.get('final_best_score'),
            'baseline_score': data.get('baseline_score'),
            'success': data.get('success', False),
            'num_turns': data.get('num_turns', len(turns_data)),
            'turns': turns_data,
        })
    return tasks


def compute_best_score_curve(turns):
    best_scores = []
    current_best = None
    for t in turns:
        s = t.get('score')
        if s is not None:
            current_best = min(current_best, s) if current_best is not None else s
        best_scores.append(current_best)
    return best_scores


def get_mean_curve(tasks, max_len=None):
    all_curves = []
    for task in tasks:
        if not task['success'] or not task['turns']:
            continue
        baseline = task.get('baseline_score')
        best_scores = compute_best_score_curve(task['turns'])
        if baseline and baseline != 0 and any(s is not None for s in best_scores):
            norm = [((baseline - s) / baseline * 100) if s is not None else None for s in best_scores]
            all_curves.append(norm)
    if not all_curves:
        return np.array([]), 0

    actual_max = max(len(c) for c in all_curves)
    if max_len:
        actual_max = min(actual_max, max_len)

    padded = []
    for curve in all_curves:
        extended = []
        last_val = 0
        for v in curve[:actual_max]:
            if v is not None:
                last_val = v
            extended.append(last_val)
        while len(extended) < actual_max:
            extended.append(last_val)
        padded.append(extended)

    return np.mean(padded, axis=0), actual_max


# Load all data
eet_tasks = load_trajectories(EET_DIR)
aide_t20_tasks = load_trajectories(AIDE_T20_DIR, 'aide')
aide_t10_tasks = load_trajectories(AIDE_T10_DIR, 'aide')
nt_tasks = load_trajectories(NO_TERM_DIR)

print(f"EET: {len(eet_tasks)}, AIDE t20: {len(aide_t20_tasks)}, AIDE t10d4: {len(aide_t10_tasks)}, No-Terminate: {len(nt_tasks)}")

# Compute mean curves
eet_mean, eet_len = get_mean_curve(eet_tasks, max_len=20)
aide20_mean, aide20_len = get_mean_curve(aide_t20_tasks, max_len=20)
aide10_mean, aide10_len = get_mean_curve(aide_t10_tasks, max_len=10)
nt_mean, nt_len = get_mean_curve(nt_tasks, max_len=20)

# Plot
fig, ax = plt.subplots(figsize=(12, 7))

if len(eet_mean) > 0:
    ax.plot(range(eet_len), eet_mean, color='#55A868', linewidth=2.5,
            label='EET (explore/exploit/terminate)', marker='o', markersize=5)
if len(nt_mean) > 0:
    ax.plot(range(nt_len), nt_mean, color='#E07B39', linewidth=2.5,
            label='EET no-terminate (explore/exploit only)', marker='D', markersize=5)
if len(aide20_mean) > 0:
    ax.plot(range(aide20_len), aide20_mean, color='#C44E52', linewidth=2.5,
            label='AIDE (t20)', marker='s', markersize=5)
if len(aide10_mean) > 0:
    ax.plot(range(aide10_len), aide10_mean, color='#8172B2', linewidth=2.5,
            label='AIDE (t10d4)', marker='^', markersize=5)

ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Turn Number')
ax.set_ylabel('Mean Improvement over Baseline (%)')
ax.set_title('Score Improvement Comparison: EET vs AIDE vs No-Terminate')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'fig5_combined_score_curves_with_nt.png')
plt.savefig(out_path, bbox_inches='tight')
plt.close()
print(f"Saved {out_path}")

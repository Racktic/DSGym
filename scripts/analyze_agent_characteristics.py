"""
Analyze and compare characteristics of React, EET, and AIDE agents.
Generates visualizations for:
1. Score progression curves (per-turn best score)
2. Action distribution analysis
3. Turn utilization / early stopping analysis
4. Iteration efficiency (score improvement per turn)
"""

import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from collections import defaultdict, Counter

# ─── Config ───────────────────────────────────────────────────────────
RESULTS_DIR = "/data/fnie/qixin/DSGym/evaluation_results"
REACT_DIR = os.path.join(RESULTS_DIR, "react_qwen3_235b_easy_v2")
EET_DIR = os.path.join(RESULTS_DIR, "eet_qwen3_235b_easy_v3")
AIDE_T20_DIR = os.path.join(RESULTS_DIR, "aide_qwen3_235b_easy_v1")
AIDE_T10_DIR = os.path.join(RESULTS_DIR, "aide_qwen3_235b_easy_t10d4")
OUTPUT_DIR = "/data/fnie/qixin/DSGym/scripts/analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})


# ─── Data Loading ─────────────────────────────────────────────────────

def load_react_trajectories():
    """Load React results and extract per-turn info from trajectory field."""
    results_file = glob.glob(os.path.join(REACT_DIR, "*_results.json"))[0]
    with open(results_file) as f:
        data = json.load(f)

    tasks = []
    for item in data:
        task_id = item['sample_id'].replace('dspredict_', '').rsplit('_', 1)[0]
        traj = item.get('trajectory', [])
        total_turns = item.get('total_turns', 0)
        success = item.get('success', False)

        # React trajectory is role-based: assistant generates code, user returns output
        # Count assistant turns as "actions"
        assistant_turns = [t for t in traj if t['role'] == 'assistant']
        # Check if last turn has done=True (early stopping)
        done = traj[-1].get('done', False) if traj else False

        tasks.append({
            'task_id': task_id,
            'total_turns': total_turns,
            'success': success,
            'done': done,
            'execution_time': item.get('execution_time', 0),
            'num_assistant_turns': len(assistant_turns),
            'trajectory': traj,
        })
    return tasks


def load_structured_trajectories(directory, agent_type):
    """Load EET or AIDE trajectory files."""
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
                'score_delta': t.get('score_delta'),
                'step_time': t.get('step_time', 0),
            }
            if agent_type == 'eet':
                if t.get('parsed_output') and t['parsed_output'].get('decision'):
                    turn_info['action'] = t['parsed_output']['decision'].get('action', 'unknown')
                else:
                    turn_info['action'] = 'unknown'
            elif agent_type == 'aide':
                turn_info['phase'] = t.get('phase', 'unknown')
            turns_data.append(turn_info)

        tasks.append({
            'task_id': data.get('task_id', data.get('challenge_name', '')),
            'agent_type': agent_type,
            'final_best_score': data.get('final_best_score'),
            'baseline_score': data.get('baseline_score'),
            'total_time': data.get('total_time', 0),
            'success': data.get('success', False),
            'num_turns': data.get('num_turns', len(turns_data)),
            'turns': turns_data,
        })
    return tasks


# ─── Analysis Functions ───────────────────────────────────────────────

def compute_best_score_curve(turns, lower_is_better=True):
    """Compute running best score across turns."""
    best_scores = []
    current_best = None
    for t in turns:
        s = t.get('score')
        if s is not None:
            if current_best is None:
                current_best = s
            else:
                if lower_is_better:
                    current_best = min(current_best, s)
                else:
                    current_best = max(current_best, s)
        best_scores.append(current_best)
    return best_scores


def normalize_score_improvement(baseline, best_scores):
    """Normalize score improvement as percentage of baseline."""
    if baseline is None or baseline == 0:
        return [None] * len(best_scores)
    return [
        ((baseline - s) / baseline * 100) if s is not None else None
        for s in best_scores
    ]


# ─── Load All Data ────────────────────────────────────────────────────
print("Loading data...")
react_tasks = load_react_trajectories()
eet_tasks = load_structured_trajectories(EET_DIR, 'eet')
aide_t20_tasks = load_structured_trajectories(AIDE_T20_DIR, 'aide')
aide_t10_tasks = load_structured_trajectories(AIDE_T10_DIR, 'aide')

print(f"React: {len(react_tasks)} tasks")
print(f"EET: {len(eet_tasks)} tasks")
print(f"AIDE t20: {len(aide_t20_tasks)} tasks")
print(f"AIDE t10d4: {len(aide_t10_tasks)} tasks")


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: Turn Distribution / Early Stopping Analysis
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Figure 1: Turn Distribution ===")

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

# React turns
react_turns = [t['total_turns'] for t in react_tasks if t['success']]
axes[0].hist(react_turns, bins=range(1, max(react_turns)+2), color='#4C72B0', alpha=0.8, edgecolor='white')
axes[0].set_title(f'React (n={len(react_turns)})')
axes[0].set_xlabel('Number of Turns')
axes[0].set_ylabel('Count')
axes[0].axvline(np.mean(react_turns), color='red', linestyle='--', label=f'Mean={np.mean(react_turns):.1f}')
axes[0].legend()

# EET turns
eet_turns = [t['num_turns'] for t in eet_tasks if t['success']]
axes[1].hist(eet_turns, bins=range(1, max(eet_turns)+2), color='#55A868', alpha=0.8, edgecolor='white')
axes[1].set_title(f'EET (n={len(eet_turns)})')
axes[1].set_xlabel('Number of Turns')
axes[1].axvline(np.mean(eet_turns), color='red', linestyle='--', label=f'Mean={np.mean(eet_turns):.1f}')
axes[1].legend()

# AIDE t20
aide20_turns = [t['num_turns'] for t in aide_t20_tasks if t['success']]
axes[2].hist(aide20_turns, bins=range(1, max(aide20_turns)+2), color='#C44E52', alpha=0.8, edgecolor='white')
axes[2].set_title(f'AIDE t20 (n={len(aide20_turns)})')
axes[2].set_xlabel('Number of Turns')
axes[2].axvline(np.mean(aide20_turns), color='red', linestyle='--', label=f'Mean={np.mean(aide20_turns):.1f}')
axes[2].legend()

# AIDE t10d4
aide10_turns = [t['num_turns'] for t in aide_t10_tasks if t['success']]
axes[3].hist(aide10_turns, bins=range(1, max(aide10_turns)+2), color='#8172B2', alpha=0.8, edgecolor='white')
axes[3].set_title(f'AIDE t10d4 (n={len(aide10_turns)})')
axes[3].set_xlabel('Number of Turns')
axes[3].axvline(np.mean(aide10_turns), color='red', linestyle='--', label=f'Mean={np.mean(aide10_turns):.1f}')
axes[3].legend()

plt.suptitle('Turn Distribution by Agent', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_turn_distribution.png'), bbox_inches='tight')
plt.close()
print("Saved fig1_turn_distribution.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: EET Action Distribution Over Turns
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Figure 2: EET Action Distribution ===")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# EET: action per turn (stacked)
max_eet_turns = max(t['num_turns'] for t in eet_tasks)
action_counts_per_turn = defaultdict(lambda: Counter())
for task in eet_tasks:
    if not task['success']:
        continue
    for t in task['turns']:
        action_counts_per_turn[t['turn']][t['action']] += 1

turns_range = range(max_eet_turns)
actions = ['explore', 'exploit', 'terminate']
colors = {'explore': '#4C72B0', 'exploit': '#55A868', 'terminate': '#C44E52'}

bottoms = np.zeros(max_eet_turns)
for action in actions:
    values = [action_counts_per_turn[i].get(action, 0) for i in turns_range]
    axes[0].bar(list(turns_range), values, bottom=bottoms, label=action,
                color=colors[action], alpha=0.85, width=0.8)
    bottoms += np.array(values)

axes[0].set_xlabel('Turn Number')
axes[0].set_ylabel('Count')
axes[0].set_title('EET: Action Distribution per Turn')
axes[0].legend()

# EET: overall action distribution
all_eet_actions = []
for task in eet_tasks:
    if not task['success']:
        continue
    for t in task['turns']:
        all_eet_actions.append(t['action'])
action_counts = Counter(all_eet_actions)
labels = list(action_counts.keys())
sizes = list(action_counts.values())
pie_colors = [colors.get(a, '#999') for a in labels]
axes[1].pie(sizes, labels=[f"{l}\n({v}, {v/sum(sizes)*100:.1f}%)" for l, v in zip(labels, sizes)],
            colors=pie_colors, autopct='', startangle=90)
axes[1].set_title('EET: Overall Action Distribution')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_eet_actions.png'), bbox_inches='tight')
plt.close()
print("Saved fig2_eet_actions.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: AIDE Phase Distribution Over Turns
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Figure 3: AIDE Phase Distribution ===")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, tasks, title in [(axes[0], aide_t20_tasks, 'AIDE t20'), (axes[1], aide_t10_tasks, 'AIDE t10d4')]:
    max_turns = max(t['num_turns'] for t in tasks)
    phase_counts_per_turn = defaultdict(lambda: Counter())
    for task in tasks:
        if not task['success']:
            continue
        for t in task['turns']:
            phase_counts_per_turn[t['turn']][t['phase']] += 1

    phases = ['draft', 'improve', 'debug', 'final_submission']
    phase_colors = {'draft': '#4C72B0', 'improve': '#55A868', 'debug': '#C44E52', 'final_submission': '#8172B2'}

    bottoms = np.zeros(max_turns)
    turns_range = range(max_turns)
    for phase in phases:
        values = [phase_counts_per_turn[i].get(phase, 0) for i in turns_range]
        ax.bar(list(turns_range), values, bottom=bottoms, label=phase,
               color=phase_colors[phase], alpha=0.85, width=0.8)
        bottoms += np.array(values)

    ax.set_xlabel('Turn Number')
    ax.set_ylabel('Count')
    ax.set_title(f'{title}: Phase Distribution per Turn')
    ax.legend(loc='upper right')

plt.suptitle('AIDE Phase Distribution Over Turns', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_aide_phases.png'), bbox_inches='tight')
plt.close()
print("Saved fig3_aide_phases.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 4: Score Progression Curves (EET & AIDE)
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Figure 4: Score Progression Curves ===")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

def plot_score_curves(ax, tasks, title, color, max_turns_limit=None):
    """Plot normalized best-score-so-far curves for each task, plus mean."""
    all_curves = []
    for task in tasks:
        if not task['success'] or not task['turns']:
            continue
        baseline = task.get('baseline_score')
        best_scores = compute_best_score_curve(task['turns'])
        if baseline and any(s is not None for s in best_scores):
            norm_curve = normalize_score_improvement(baseline, best_scores)
            all_curves.append(norm_curve)

    if not all_curves:
        ax.set_title(f'{title} (no data)')
        return

    # Pad curves to max length
    max_len = max(len(c) for c in all_curves)
    if max_turns_limit:
        max_len = min(max_len, max_turns_limit)

    # Plot individual curves (light)
    for curve in all_curves:
        # Extend last value to max_len
        extended = []
        last_val = None
        for v in curve[:max_len]:
            if v is not None:
                last_val = v
            extended.append(last_val)
        while len(extended) < max_len:
            extended.append(last_val)
        ax.plot(range(max_len), extended, color=color, alpha=0.12, linewidth=0.8)

    # Compute and plot mean curve
    padded = []
    for curve in all_curves:
        extended = []
        last_val = 0
        for v in curve[:max_len]:
            if v is not None:
                last_val = v
            extended.append(last_val)
        while len(extended) < max_len:
            extended.append(last_val)
        padded.append(extended)

    mean_curve = np.mean(padded, axis=0)
    ax.plot(range(max_len), mean_curve, color=color, linewidth=2.5, label='Mean')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Turn Number')
    ax.set_ylabel('Improvement over Baseline (%)')
    ax.set_title(title)
    ax.legend()

plot_score_curves(axes[0], eet_tasks, 'EET', '#55A868')
plot_score_curves(axes[1], aide_t20_tasks, 'AIDE (t20)', '#C44E52')
plot_score_curves(axes[2], aide_t10_tasks, 'AIDE (t10d4)', '#8172B2')

plt.suptitle('Score Improvement over Baseline per Turn', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_score_progression.png'), bbox_inches='tight')
plt.close()
print("Saved fig4_score_progression.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 5: Combined Mean Score Curve Comparison
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Figure 5: Combined Score Curves ===")

fig, ax = plt.subplots(figsize=(10, 6))

def get_mean_curve(tasks, max_len=None):
    """Get mean normalized improvement curve."""
    all_curves = []
    for task in tasks:
        if not task['success'] or not task['turns']:
            continue
        baseline = task.get('baseline_score')
        best_scores = compute_best_score_curve(task['turns'])
        if baseline and any(s is not None for s in best_scores):
            norm_curve = normalize_score_improvement(baseline, best_scores)
            all_curves.append(norm_curve)
    if not all_curves:
        return [], 0

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

eet_mean, eet_len = get_mean_curve(eet_tasks)
aide20_mean, aide20_len = get_mean_curve(aide_t20_tasks)
aide10_mean, aide10_len = get_mean_curve(aide_t10_tasks)

if len(eet_mean) > 0:
    ax.plot(range(eet_len), eet_mean, color='#55A868', linewidth=2.5, label='EET', marker='o', markersize=4)
if len(aide20_mean) > 0:
    ax.plot(range(aide20_len), aide20_mean, color='#C44E52', linewidth=2.5, label='AIDE (t20)', marker='s', markersize=4)
if len(aide10_mean) > 0:
    ax.plot(range(aide10_len), aide10_mean, color='#8172B2', linewidth=2.5, label='AIDE (t10d4)', marker='^', markersize=4)

ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Turn Number')
ax.set_ylabel('Mean Improvement over Baseline (%)')
ax.set_title('Score Improvement Comparison: EET vs AIDE')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_combined_score_curves.png'), bbox_inches='tight')
plt.close()
print("Saved fig5_combined_score_curves.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 6: React Early Stopping Analysis
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Figure 6: React Early Stopping ===")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Check max turns setting - React agent probably has max_turns=20
react_max_turns = 20  # default
react_success_turns = [t['total_turns'] for t in react_tasks if t['success']]
react_fail_turns = [t['total_turns'] for t in react_tasks if not t['success']]

# Early stop vs max-turn
early_stopped = sum(1 for t in react_success_turns if t < react_max_turns)
hit_max = sum(1 for t in react_success_turns if t >= react_max_turns)

axes[0].bar(['Early Stopped\n(< max turns)', 'Hit Max Turns'],
            [early_stopped, hit_max],
            color=['#55A868', '#C44E52'], alpha=0.8)
axes[0].set_ylabel('Count')
axes[0].set_title(f'React: Early Stopping (max={react_max_turns})')
for i, v in enumerate([early_stopped, hit_max]):
    axes[0].text(i, v + 0.3, str(v), ha='center', fontweight='bold')

# Compare with EET terminate
eet_terminated = sum(1 for task in eet_tasks if task['success'] and
                     any(t['action'] == 'terminate' for t in task['turns']))
eet_no_terminate = sum(1 for task in eet_tasks if task['success']) - eet_terminated

axes[1].bar(['Has Terminate\nAction', 'No Terminate\n(hit max turns)'],
            [eet_terminated, eet_no_terminate],
            color=['#55A868', '#C44E52'], alpha=0.8)
axes[1].set_ylabel('Count')
axes[1].set_title('EET: Terminate Usage')
for i, v in enumerate([eet_terminated, eet_no_terminate]):
    axes[1].text(i, v + 0.3, str(v), ha='center', fontweight='bold')

plt.suptitle('Early Stopping Behavior', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_early_stopping.png'), bbox_inches='tight')
plt.close()
print("Saved fig6_early_stopping.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 7: Iteration Efficiency — Score Improvement at Key Turns
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Figure 7: Iteration Efficiency ===")

fig, ax = plt.subplots(figsize=(10, 6))

def get_improvement_at_turns(tasks, checkpoints):
    """Get mean improvement at specific turn checkpoints."""
    improvements = {cp: [] for cp in checkpoints}
    for task in tasks:
        if not task['success'] or not task['turns']:
            continue
        baseline = task.get('baseline_score')
        if not baseline:
            continue
        best_scores = compute_best_score_curve(task['turns'])
        for cp in checkpoints:
            if cp < len(best_scores) and best_scores[cp] is not None:
                imp = (baseline - best_scores[cp]) / baseline * 100
                improvements[cp].append(imp)
            elif best_scores:
                # Use last available
                last = None
                for s in reversed(best_scores):
                    if s is not None:
                        last = s
                        break
                if last is not None:
                    imp = (baseline - last) / baseline * 100
                    improvements[cp].append(imp)
    return {cp: np.mean(v) if v else 0 for cp, v in improvements.items()}

checkpoints = [2, 4, 6, 8, 10, 12, 14, 16, 18]

eet_imp = get_improvement_at_turns(eet_tasks, checkpoints)
aide20_imp = get_improvement_at_turns(aide_t20_tasks, checkpoints)
aide10_imp = get_improvement_at_turns(aide_t10_tasks, checkpoints[:5])  # only up to 10

x_eet = [cp for cp in checkpoints if cp in eet_imp]
y_eet = [eet_imp[cp] for cp in x_eet]
x_aide20 = [cp for cp in checkpoints if cp in aide20_imp]
y_aide20 = [aide20_imp[cp] for cp in x_aide20]
x_aide10 = [cp for cp in checkpoints[:5] if cp in aide10_imp]
y_aide10 = [aide10_imp[cp] for cp in x_aide10]

ax.plot(x_eet, y_eet, color='#55A868', linewidth=2.5, label='EET', marker='o', markersize=6)
ax.plot(x_aide20, y_aide20, color='#C44E52', linewidth=2.5, label='AIDE (t20)', marker='s', markersize=6)
ax.plot(x_aide10, y_aide10, color='#8172B2', linewidth=2.5, label='AIDE (t10d4)', marker='^', markersize=6)

ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Turn Number')
ax.set_ylabel('Mean Improvement over Baseline (%)')
ax.set_title('Iteration Efficiency: Improvement at Checkpoints')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig7_iteration_efficiency.png'), bbox_inches='tight')
plt.close()
print("Saved fig7_iteration_efficiency.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 8: EET Action Transition Analysis
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Figure 8: EET Exploration vs Exploitation Phase ===")

fig, ax = plt.subplots(figsize=(10, 5))

# For each EET task, compute when exploration ends and exploitation begins
explore_ratios = []
exploit_ratios = []
terminate_turns = []

for task in eet_tasks:
    if not task['success']:
        continue
    actions = [t['action'] for t in task['turns']]
    n = len(actions)
    if n == 0:
        continue

    n_explore = sum(1 for a in actions if a == 'explore')
    n_exploit = sum(1 for a in actions if a == 'exploit')
    n_terminate = sum(1 for a in actions if a == 'terminate')

    explore_ratios.append(n_explore / n * 100)
    exploit_ratios.append(n_exploit / n * 100)

    # Find first exploit turn
    first_exploit = next((i for i, a in enumerate(actions) if a == 'exploit'), n)
    terminate_turns.append(n)

# Stacked bar: explore vs exploit ratio per task (sorted by explore ratio)
sorted_indices = np.argsort(explore_ratios)[::-1]
explore_sorted = [explore_ratios[i] for i in sorted_indices]
exploit_sorted = [exploit_ratios[i] for i in sorted_indices]
terminate_sorted = [100 - explore_sorted[i] - exploit_sorted[i] for i in range(len(sorted_indices))]

x = range(len(sorted_indices))
ax.bar(x, explore_sorted, color='#4C72B0', label='Explore', alpha=0.85)
ax.bar(x, exploit_sorted, bottom=explore_sorted, color='#55A868', label='Exploit', alpha=0.85)
ax.bar(x, terminate_sorted,
       bottom=[e + x for e, x in zip(explore_sorted, exploit_sorted)],
       color='#C44E52', label='Terminate', alpha=0.85)

ax.set_xlabel('Task (sorted by explore ratio)')
ax.set_ylabel('Action Ratio (%)')
ax.set_title('EET: Explore/Exploit/Terminate Ratio per Task')
ax.legend()
ax.set_xlim(-0.5, len(sorted_indices)-0.5)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig8_eet_action_ratios.png'), bbox_inches='tight')
plt.close()
print("Saved fig8_eet_action_ratios.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 9: AIDE Phase Ratio Analysis
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Figure 9: AIDE Phase Ratios ===")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, tasks, title in [(axes[0], aide_t20_tasks, 'AIDE t20'), (axes[1], aide_t10_tasks, 'AIDE t10d4')]:
    phase_ratios = {'draft': [], 'improve': [], 'debug': [], 'final_submission': []}
    for task in tasks:
        if not task['success']:
            continue
        phases = [t['phase'] for t in task['turns']]
        n = len(phases)
        if n == 0:
            continue
        for p in phase_ratios:
            phase_ratios[p].append(sum(1 for x in phases if x == p) / n * 100)

    # Box plot
    phase_data = [phase_ratios[p] for p in ['draft', 'improve', 'debug', 'final_submission']]
    bp = ax.boxplot(phase_data, labels=['Draft', 'Improve', 'Debug', 'Final\nSubmit'],
                    patch_artist=True)
    phase_colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    for patch, color in zip(bp['boxes'], phase_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Phase Ratio (%)')
    ax.set_title(f'{title}: Phase Distribution')
    # Add mean annotations
    for i, p in enumerate(['draft', 'improve', 'debug', 'final_submission']):
        if phase_ratios[p]:
            mean_val = np.mean(phase_ratios[p])
            ax.annotate(f'{mean_val:.1f}%', (i+1, mean_val), textcoords="offset points",
                       xytext=(15, 5), fontsize=9, color='red')

plt.suptitle('AIDE Phase Distribution Across Tasks', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig9_aide_phase_ratios.png'), bbox_inches='tight')
plt.close()
print("Saved fig9_aide_phase_ratios.png")


# ═══════════════════════════════════════════════════════════════════════
# Print Summary Statistics
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

# React early stopping
print(f"\n--- React ---")
print(f"Total tasks: {len(react_tasks)}, Successful: {sum(1 for t in react_tasks if t['success'])}")
print(f"Turn distribution: min={min(react_success_turns)}, max={max(react_success_turns)}, "
      f"mean={np.mean(react_success_turns):.1f}, median={np.median(react_success_turns):.0f}")
print(f"Early stopped (<{react_max_turns}): {early_stopped}/{len(react_success_turns)} "
      f"({early_stopped/len(react_success_turns)*100:.1f}%)")
print(f"Hit max turns: {hit_max}/{len(react_success_turns)} "
      f"({hit_max/len(react_success_turns)*100:.1f}%)")

# EET
print(f"\n--- EET ---")
print(f"Total tasks: {len(eet_tasks)}, Successful: {sum(1 for t in eet_tasks if t['success'])}")
print(f"Turn distribution: min={min(eet_turns)}, max={max(eet_turns)}, "
      f"mean={np.mean(eet_turns):.1f}, median={np.median(eet_turns):.0f}")
print(f"Used terminate action: {eet_terminated}/{sum(1 for t in eet_tasks if t['success'])} "
      f"({eet_terminated/sum(1 for t in eet_tasks if t['success'])*100:.1f}%)")
# Action distribution
all_actions = Counter()
for task in eet_tasks:
    if task['success']:
        for t in task['turns']:
            all_actions[t['action']] += 1
total_actions = sum(all_actions.values())
print(f"Action distribution: " + ", ".join(f"{a}={c} ({c/total_actions*100:.1f}%)" for a, c in all_actions.most_common()))

# EET: avg explore turns before first exploit
first_exploit_turns = []
for task in eet_tasks:
    if not task['success']:
        continue
    actions = [t['action'] for t in task['turns']]
    first_exploit = next((i for i, a in enumerate(actions) if a in ('exploit', 'terminate')), len(actions))
    first_exploit_turns.append(first_exploit)
print(f"Avg turns before first exploit/terminate: {np.mean(first_exploit_turns):.1f}")

# AIDE t20
print(f"\n--- AIDE t20 ---")
aide20_success = [t for t in aide_t20_tasks if t['success']]
print(f"Total tasks: {len(aide_t20_tasks)}, Successful: {len(aide20_success)}")
print(f"Turn distribution: min={min(aide20_turns)}, max={max(aide20_turns)}, "
      f"mean={np.mean(aide20_turns):.1f}")
# Phase counts
aide20_phases = Counter()
for task in aide20_success:
    for t in task['turns']:
        aide20_phases[t['phase']] += 1
total_phases = sum(aide20_phases.values())
print(f"Phase distribution: " + ", ".join(f"{p}={c} ({c/total_phases*100:.1f}%)" for p, c in aide20_phases.most_common()))

# When does score first improve?
first_improve_turn = []
for task in aide20_success:
    baseline = task.get('baseline_score')
    if not baseline:
        continue
    for i, t in enumerate(task['turns']):
        if t['score'] is not None and t['score'] < baseline:
            first_improve_turn.append(i)
            break
    else:
        first_improve_turn.append(task['num_turns'])
print(f"Avg turn of first score improvement: {np.mean(first_improve_turn):.1f}")

# Score stagnation analysis
stagnant_turns = []
for task in aide20_success:
    best = None
    last_improve = 0
    for i, t in enumerate(task['turns']):
        if t['score'] is not None:
            if best is None or t['score'] < best:
                best = t['score']
                last_improve = i
    stagnant = task['num_turns'] - 1 - last_improve
    stagnant_turns.append(stagnant)
print(f"Avg stagnant turns at end (no improvement): {np.mean(stagnant_turns):.1f}")

# AIDE t10
print(f"\n--- AIDE t10d4 ---")
aide10_success = [t for t in aide_t10_tasks if t['success']]
print(f"Total tasks: {len(aide_t10_tasks)}, Successful: {len(aide10_success)}")
aide10_phases = Counter()
for task in aide10_success:
    for t in task['turns']:
        aide10_phases[t['phase']] += 1
total_phases10 = sum(aide10_phases.values())
print(f"Phase distribution: " + ", ".join(f"{p}={c} ({c/total_phases10*100:.1f}%)" for p, c in aide10_phases.most_common()))

print(f"\n{'='*70}")
print("All figures saved to:", OUTPUT_DIR)

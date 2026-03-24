"""
Compare EET v3 (with terminate) vs EET no-terminate (explore/exploit only).

Generates:
- Fig NT-A: Combined score curves (v3 vs no-terminate)
- Fig NT-B: Per-task percentile comparison (public + private leaderboard)
- Fig NT-C: Turn distribution comparison
- Fig NT-D: Action distribution analysis
- Fig NT-E: Wasted turns & efficiency
- Fig NT-F: Per-task score progression heatmap
"""

import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

# ─── Config ───────────────────────────────────────────────────────────
RESULTS_DIR = "/data/fnie/qixin/DSGym/evaluation_results"
EET_V3_DIR = os.path.join(RESULTS_DIR, "eet_qwen3_235b_easy_v3")
NO_TERM_DIR = os.path.join(RESULTS_DIR, "eet_no_terminate_full")
OUTPUT_DIR = "/data/fnie/qixin/DSGym/scripts/analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

V3_COLOR = '#55A868'
NT_COLOR = '#E07B39'
V3_LABEL = 'EET v3 (with terminate)'
NT_LABEL = 'EET no-terminate'


# ─── Data Loading ─────────────────────────────────────────────────────

def load_trajectories(directory):
    """Load EET trajectory files."""
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
                'parse_success': t.get('parse_success', False),
            }
            if t.get('parsed_output') and t['parsed_output'].get('decision'):
                turn_info['action'] = t['parsed_output']['decision'].get('action', 'unknown')
            else:
                turn_info['action'] = 'unknown'
            turns_data.append(turn_info)
        tasks.append({
            'task_id': data.get('task_id', data.get('challenge_name', '')),
            'challenge_name': data.get('challenge_name', ''),
            'final_best_score': data.get('final_best_score'),
            'baseline_score': data.get('baseline_score'),
            'total_time': data.get('total_time', 0),
            'success': data.get('success', False),
            'num_turns': data.get('num_turns', len(turns_data)),
            'turns': turns_data,
        })
    return tasks


def load_results(directory):
    """Load results JSON with Kaggle submission info."""
    results_files = glob.glob(os.path.join(directory, "*_results.json"))
    if not results_files:
        return {}
    with open(results_files[0]) as f:
        data = json.load(f)
    out = {}
    for r in data:
        name = r.get("extra_info", {}).get("metadata_id", "?")
        kaggle = r.get("metrics", {}).get("kaggle_submission", {})
        details = kaggle.get("details", {})
        lb_stats = details.get("leaderboard_stats", {})

        # Compute private percentile if we have data
        private_pct = None
        private_score = details.get("private_score")
        private_scores = lb_stats.get("private_scores", [])
        if private_score and private_score != '' and private_scores:
            try:
                ps = float(private_score)
                better_count = sum(1 for s in private_scores if s > ps)
                private_pct = better_count / len(private_scores) * 100
            except (ValueError, TypeError):
                pass

        out[name] = {
            "public_score": details.get("public_score"),
            "private_score": private_score,
            "public_percentile": details.get("public_percentile"),
            "private_percentile": private_pct,
            "public_medal": details.get("public_medal", "none"),
            "valid_submission": kaggle.get("success", False) and not kaggle.get("skipped", False),
            "total_turns": r.get("total_turns", 0),
            "leaderboard_stats": lb_stats,
        }
    return out


# ─── Score Curve Helpers ──────────────────────────────────────────────

def compute_best_score_curve(turns, lower_is_better=True):
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
    if baseline is None or baseline == 0:
        return [None] * len(best_scores)
    return [
        ((baseline - s) / baseline * 100) if s is not None else None
        for s in best_scores
    ]


def get_mean_curve(tasks, max_len=None):
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


def get_all_curves_padded(tasks, max_len=None):
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

    return padded, actual_max


# ─── Load Data ────────────────────────────────────────────────────────
print("Loading data...")
v3_tasks = load_trajectories(EET_V3_DIR)
nt_tasks = load_trajectories(NO_TERM_DIR)
v3_results = load_results(EET_V3_DIR)
nt_results = load_results(NO_TERM_DIR)
print(f"EET v3: {len(v3_tasks)} trajectories, {len(v3_results)} results")
print(f"No-terminate: {len(nt_tasks)} trajectories, {len(nt_results)} results")


# ═══════════════════════════════════════════════════════════════════════
# Figure NT-A: Combined Score Curves
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Figure NT-A: Combined Score Curves ===")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: individual + mean curves
ax = axes[0]
v3_curves, v3_len = get_all_curves_padded(v3_tasks, max_len=20)
nt_curves, nt_len = get_all_curves_padded(nt_tasks, max_len=20)

for curve in v3_curves:
    ax.plot(range(len(curve)), curve, color=V3_COLOR, alpha=0.08, linewidth=0.7)
for curve in nt_curves:
    ax.plot(range(len(curve)), curve, color=NT_COLOR, alpha=0.08, linewidth=0.7)

v3_mean, v3_mlen = get_mean_curve(v3_tasks, max_len=20)
nt_mean, nt_mlen = get_mean_curve(nt_tasks, max_len=20)

if len(v3_mean) > 0:
    ax.plot(range(v3_mlen), v3_mean, color=V3_COLOR, linewidth=2.5,
            label=V3_LABEL, marker='o', markersize=5)
if len(nt_mean) > 0:
    ax.plot(range(nt_mlen), nt_mean, color=NT_COLOR, linewidth=2.5,
            label=NT_LABEL, marker='s', markersize=5)

ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Turn Number')
ax.set_ylabel('Mean Improvement over Baseline (%)')
ax.set_title('Score Improvement Curves (Individual + Mean)')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# Right: mean curves with std band
ax = axes[1]
if len(v3_mean) > 0:
    v3_arr = np.array([c[:v3_mlen] for c in v3_curves if len(c) >= v3_mlen])
    if len(v3_arr) > 0:
        v3_std = np.std(v3_arr, axis=0)
        ax.fill_between(range(v3_mlen), v3_mean - v3_std, v3_mean + v3_std,
                         color=V3_COLOR, alpha=0.15)
    ax.plot(range(v3_mlen), v3_mean, color=V3_COLOR, linewidth=2.5,
            label=V3_LABEL, marker='o', markersize=5)
if len(nt_mean) > 0:
    nt_arr = np.array([c[:nt_mlen] for c in nt_curves if len(c) >= nt_mlen])
    if len(nt_arr) > 0:
        nt_std = np.std(nt_arr, axis=0)
        ax.fill_between(range(nt_mlen), nt_mean - nt_std, nt_mean + nt_std,
                         color=NT_COLOR, alpha=0.15)
    ax.plot(range(nt_mlen), nt_mean, color=NT_COLOR, linewidth=2.5,
            label=NT_LABEL, marker='s', markersize=5)

ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Turn Number')
ax.set_ylabel('Mean Improvement over Baseline (%)')
ax.set_title('Mean Score Curves with Std Band')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

plt.suptitle('EET v3 vs No-Terminate: Score Improvement Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figNT_A_score_curves.png'), bbox_inches='tight')
plt.close()
print("Saved figNT_A_score_curves.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure NT-B: Per-task Percentile Comparison (Public + Private)
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Figure NT-B: Percentile Comparison ===")

all_task_names = sorted(set(list(v3_results.keys()) + list(nt_results.keys())))

# Collect paired percentile data
paired_public = []
paired_private = []
paired_names = []

for name in all_task_names:
    r3 = v3_results.get(name, {})
    rn = nt_results.get(name, {})
    p3_pub = r3.get("public_percentile")
    pn_pub = rn.get("public_percentile")
    p3_prv = r3.get("private_percentile")
    pn_prv = rn.get("private_percentile")

    if p3_pub is not None and pn_pub is not None:
        paired_public.append((p3_pub, pn_pub))
        paired_private.append((p3_prv, pn_prv))
        paired_names.append(name.replace("playground-series-", "ps-"))

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Top-left: Public percentile scatter
ax = axes[0][0]
if paired_public:
    x = [p[0] for p in paired_public]
    y = [p[1] for p in paired_public]
    colors_scatter = ['#55A868' if yi > xi else '#C44E52' if yi < xi else '#888'
                      for xi, yi in zip(x, y)]
    ax.scatter(x, y, c=colors_scatter, s=60, alpha=0.8, edgecolors='white', linewidth=0.5, zorder=5)
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Equal')
    ax.set_xlabel('EET v3 Percentile')
    ax.set_ylabel('No-Terminate Percentile')
    ax.set_title(f'Public LB: Per-Task Percentile (n={len(paired_public)})')

    for i, (xi, yi) in enumerate(zip(x, y)):
        if abs(yi - xi) > 20:
            ax.annotate(paired_names[i], (xi, yi), fontsize=7, alpha=0.7,
                        xytext=(5, 5), textcoords='offset points')

    better = sum(1 for xi, yi in zip(x, y) if yi > xi)
    worse = sum(1 for xi, yi in zip(x, y) if yi < xi)
    same = len(x) - better - worse
    avg_diff = np.mean([yi - xi for xi, yi in zip(x, y)])
    textstr = f'NT better: {better}\nNT worse: {worse}\nSame: {same}\nAvg diff: {avg_diff:+.1f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend(loc='lower right')
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

# Top-right: Public percentile bar chart (sorted by diff)
ax = axes[0][1]
if paired_public:
    diffs = [(paired_names[i], paired_public[i][1] - paired_public[i][0])
             for i in range(len(paired_public))]
    diffs.sort(key=lambda x: x[1])
    names_sorted = [d[0] for d in diffs]
    vals_sorted = [d[1] for d in diffs]
    bar_colors = ['#55A868' if v > 0 else '#C44E52' for v in vals_sorted]
    ax.barh(range(len(names_sorted)), vals_sorted, color=bar_colors, alpha=0.8)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=6)
    ax.set_xlabel('Percentile Change (NT - v3)')
    ax.set_title('Public LB: Per-Task Percentile Change')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')

# Bottom-left: Private percentile scatter
ax = axes[1][0]
paired_prv_valid = [(p3, pn) for (p3, pn) in paired_private if p3 is not None and pn is not None]
paired_prv_names = [paired_names[i] for i, (p3, pn) in enumerate(paired_private) if p3 is not None and pn is not None]
if paired_prv_valid:
    x = [p[0] for p in paired_prv_valid]
    y = [p[1] for p in paired_prv_valid]
    colors_scatter = ['#55A868' if yi > xi else '#C44E52' if yi < xi else '#888'
                      for xi, yi in zip(x, y)]
    ax.scatter(x, y, c=colors_scatter, s=60, alpha=0.8, edgecolors='white', linewidth=0.5, zorder=5)
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Equal')
    ax.set_xlabel('EET v3 Percentile')
    ax.set_ylabel('No-Terminate Percentile')
    ax.set_title(f'Private LB: Per-Task Percentile (n={len(paired_prv_valid)})')

    better = sum(1 for xi, yi in zip(x, y) if yi > xi)
    worse = sum(1 for xi, yi in zip(x, y) if yi < xi)
    avg_diff = np.mean([yi - xi for xi, yi in zip(x, y)])
    textstr = f'NT better: {better}\nNT worse: {worse}\nAvg diff: {avg_diff:+.1f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend(loc='lower right')
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'No private LB data', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Private LB: Per-Task Percentile')

# Bottom-right: Private percentile bar chart
ax = axes[1][1]
if paired_prv_valid:
    diffs = [(paired_prv_names[i], paired_prv_valid[i][1] - paired_prv_valid[i][0])
             for i in range(len(paired_prv_valid))]
    diffs.sort(key=lambda x: x[1])
    names_sorted = [d[0] for d in diffs]
    vals_sorted = [d[1] for d in diffs]
    bar_colors = ['#55A868' if v > 0 else '#C44E52' for v in vals_sorted]
    ax.barh(range(len(names_sorted)), vals_sorted, color=bar_colors, alpha=0.8)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=6)
    ax.set_xlabel('Percentile Change (NT - v3)')
    ax.set_title('Private LB: Per-Task Percentile Change')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')
else:
    ax.text(0.5, 0.5, 'No private LB data', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Private LB: Per-Task Percentile Change')

plt.suptitle('EET v3 vs No-Terminate: Leaderboard Percentile Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figNT_B_percentile.png'), bbox_inches='tight')
plt.close()
print("Saved figNT_B_percentile.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure NT-C: Turn Distribution Comparison
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Figure NT-C: Turn Distribution ===")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Left: v3 turn histogram
v3_turns = [t['num_turns'] for t in v3_tasks if t['success']]
nt_turns = [t['num_turns'] for t in nt_tasks if t['success']]

ax = axes[0]
max_t = max(max(v3_turns, default=1), max(nt_turns, default=1))
bins = range(1, max_t + 2)
ax.hist(v3_turns, bins=bins, alpha=0.6, color=V3_COLOR, label=f'v3 (mean={np.mean(v3_turns):.1f})', edgecolor='white')
ax.hist(nt_turns, bins=bins, alpha=0.6, color=NT_COLOR, label=f'NT (mean={np.mean(nt_turns):.1f})', edgecolor='white')
ax.set_xlabel('Number of Turns')
ax.set_ylabel('Count')
ax.set_title('Turn Distribution Comparison')
ax.legend()

# Middle: v3 terminate behavior
ax = axes[1]
v3_terminated = sum(1 for task in v3_tasks if task['success'] and
                     any(t['action'] == 'terminate' for t in task['turns']))
v3_no_terminate = sum(1 for task in v3_tasks if task['success']) - v3_terminated

ax.bar(['Has Terminate\nAction', 'No Terminate\n(hit max turns)'],
        [v3_terminated, v3_no_terminate],
        color=[V3_COLOR, '#C44E52'], alpha=0.8)
ax.set_ylabel('Count')
ax.set_title('v3: Terminate Usage')
for i, v in enumerate([v3_terminated, v3_no_terminate]):
    ax.text(i, v + 0.3, str(v), ha='center', fontweight='bold')

# Right: Time comparison (box plot)
ax = axes[2]
v3_times = [t['total_time'] / 60 for t in v3_tasks if t['success']]
nt_times = [t['total_time'] / 60 for t in nt_tasks if t['success']]
bp = ax.boxplot([v3_times, nt_times], labels=['v3', 'No-Terminate'], patch_artist=True)
bp['boxes'][0].set_facecolor(V3_COLOR)
bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor(NT_COLOR)
bp['boxes'][1].set_alpha(0.7)
ax.set_ylabel('Time (minutes)')
ax.set_title('Execution Time Comparison')
v3_mean_time = np.mean(v3_times)
nt_mean_time = np.mean(nt_times)
ax.annotate(f'Mean: {v3_mean_time:.1f}m', (1, v3_mean_time),
            textcoords="offset points", xytext=(20, 0), fontsize=9, color='red')
ax.annotate(f'Mean: {nt_mean_time:.1f}m', (2, nt_mean_time),
            textcoords="offset points", xytext=(20, 0), fontsize=9, color='red')

plt.suptitle('EET v3 vs No-Terminate: Turn Distribution & Timing', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figNT_C_turns.png'), bbox_inches='tight')
plt.close()
print("Saved figNT_C_turns.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure NT-D: Action Distribution Analysis
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Figure NT-D: Action Distribution ===")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Helper to extract action counts per turn
def get_action_counts_per_turn(tasks, max_turns=20):
    counts = defaultdict(lambda: Counter())
    for task in tasks:
        if not task['success']:
            continue
        for t in task['turns']:
            if t['turn'] < max_turns:
                counts[t['turn']][t['action']] += 1
    return counts

# Top-left: v3 action distribution per turn (stacked)
ax = axes[0][0]
v3_act_per_turn = get_action_counts_per_turn(v3_tasks)
max_turn_v3 = max(v3_act_per_turn.keys()) + 1 if v3_act_per_turn else 20
actions_v3 = ['explore', 'exploit', 'terminate']
colors_v3 = {'explore': '#4C72B0', 'exploit': '#55A868', 'terminate': '#C44E52', 'unknown': '#999'}

bottoms = np.zeros(max_turn_v3)
for action in actions_v3:
    values = [v3_act_per_turn[i].get(action, 0) for i in range(max_turn_v3)]
    ax.bar(range(max_turn_v3), values, bottom=bottoms, label=action,
            color=colors_v3.get(action, '#999'), alpha=0.85, width=0.8)
    bottoms += np.array(values)
ax.set_xlabel('Turn Number')
ax.set_ylabel('Count')
ax.set_title('v3: Action Distribution per Turn')
ax.legend()

# Top-right: no-terminate action distribution per turn
ax = axes[0][1]
nt_act_per_turn = get_action_counts_per_turn(nt_tasks)
max_turn_nt = max(nt_act_per_turn.keys()) + 1 if nt_act_per_turn else 20
actions_nt = ['explore', 'exploit']

bottoms = np.zeros(max_turn_nt)
for action in actions_nt:
    values = [nt_act_per_turn[i].get(action, 0) for i in range(max_turn_nt)]
    ax.bar(range(max_turn_nt), values, bottom=bottoms, label=action,
            color=colors_v3.get(action, '#999'), alpha=0.85, width=0.8)
    bottoms += np.array(values)
# Also show unknown/terminate if any leaked through
for action in ['terminate', 'unknown']:
    values = [nt_act_per_turn[i].get(action, 0) for i in range(max_turn_nt)]
    if sum(values) > 0:
        ax.bar(range(max_turn_nt), values, bottom=bottoms, label=action,
                color=colors_v3.get(action, '#999'), alpha=0.85, width=0.8)
        bottoms += np.array(values)
ax.set_xlabel('Turn Number')
ax.set_ylabel('Count')
ax.set_title('No-Terminate: Action Distribution per Turn')
ax.legend()

# Bottom-left: Overall action ratio comparison (pie charts side by side)
ax = axes[1][0]
v3_all_actions = Counter()
nt_all_actions = Counter()
for task in v3_tasks:
    if task['success']:
        for t in task['turns']:
            v3_all_actions[t['action']] += 1
for task in nt_tasks:
    if task['success']:
        for t in task['turns']:
            nt_all_actions[t['action']] += 1

# Bar chart comparison
all_actions_set = sorted(set(list(v3_all_actions.keys()) + list(nt_all_actions.keys())))
x_pos = np.arange(len(all_actions_set))
width = 0.35
v3_counts = [v3_all_actions.get(a, 0) for a in all_actions_set]
nt_counts = [nt_all_actions.get(a, 0) for a in all_actions_set]
v3_total = sum(v3_counts)
nt_total = sum(nt_counts)
v3_pcts = [c/v3_total*100 for c in v3_counts]
nt_pcts = [c/nt_total*100 for c in nt_counts]

bars1 = ax.bar(x_pos - width/2, v3_pcts, width, label='v3', color=V3_COLOR, alpha=0.8)
bars2 = ax.bar(x_pos + width/2, nt_pcts, width, label='No-Terminate', color=NT_COLOR, alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(all_actions_set)
ax.set_ylabel('Percentage (%)')
ax.set_title('Overall Action Distribution')
ax.legend()
for bar_group in [bars1, bars2]:
    for bar in bar_group:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                    f'{h:.1f}%', ha='center', va='bottom', fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# Bottom-right: Per-task explore/exploit ratio (no-terminate)
ax = axes[1][1]
nt_explore_ratios = []
nt_task_names = []
for task in nt_tasks:
    if not task['success']:
        continue
    actions = [t['action'] for t in task['turns']]
    n = len(actions)
    if n == 0:
        continue
    n_explore = sum(1 for a in actions if a == 'explore')
    nt_explore_ratios.append(n_explore / n * 100)
    nt_task_names.append(task['task_id'].replace("playground-series-", "ps-"))

sorted_indices = np.argsort(nt_explore_ratios)[::-1]
explore_sorted = [nt_explore_ratios[i] for i in sorted_indices]
exploit_sorted = [100 - nt_explore_ratios[i] for i in sorted_indices]
names_sorted = [nt_task_names[i] for i in sorted_indices]

x = range(len(sorted_indices))
ax.bar(x, explore_sorted, color='#4C72B0', label='Explore', alpha=0.85)
ax.bar(x, exploit_sorted, bottom=explore_sorted, color='#55A868', label='Exploit', alpha=0.85)
ax.set_xlabel('Task (sorted by explore ratio)')
ax.set_ylabel('Action Ratio (%)')
ax.set_title('No-Terminate: Explore/Exploit Ratio per Task')
ax.legend()
ax.set_xlim(-0.5, len(sorted_indices)-0.5)

plt.suptitle('EET v3 vs No-Terminate: Action Distribution Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figNT_D_actions.png'), bbox_inches='tight')
plt.close()
print("Saved figNT_D_actions.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure NT-E: Wasted Turns & Efficiency
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Figure NT-E: Wasted Turns & Efficiency ===")

def compute_wasted_turns(tasks):
    wasted_per_task = []
    best_turn_positions = []
    total_turns_all = 0
    total_wasted = 0

    for task in tasks:
        if not task['success'] or not task['turns']:
            continue
        scores = [t.get('score') for t in task['turns']]
        valid = [(i, s) for i, s in enumerate(scores) if s is not None]
        if not valid:
            continue

        baseline = task.get('baseline_score')
        final = task.get('final_best_score')
        lower = (final < baseline) if (baseline and final) else True

        if lower:
            best_val = min(s for _, s in valid)
        else:
            best_val = max(s for _, s in valid)
        best_idx = next(i for i, s in valid if s == best_val)

        n_turns = len(task['turns'])
        wasted = n_turns - best_idx - 1
        wasted_per_task.append(wasted / n_turns * 100)
        total_turns_all += n_turns
        total_wasted += wasted
        best_turn_positions.append(best_idx)

    return wasted_per_task, total_turns_all, total_wasted, best_turn_positions

v3_wasted, v3_total, v3_total_wasted, v3_best_pos = compute_wasted_turns(v3_tasks)
nt_wasted, nt_total, nt_total_wasted, nt_best_pos = compute_wasted_turns(nt_tasks)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Left: Wasted turns distribution
ax = axes[0]
bins = np.arange(0, 105, 10)
ax.hist(v3_wasted, bins=bins, alpha=0.6, color=V3_COLOR,
        label=f'v3 (mean={np.mean(v3_wasted):.1f}%)', edgecolor='white')
ax.hist(nt_wasted, bins=bins, alpha=0.6, color=NT_COLOR,
        label=f'NT (mean={np.mean(nt_wasted):.1f}%)', edgecolor='white')
ax.set_xlabel('Wasted Turns (%)')
ax.set_ylabel('Count')
ax.set_title('Distribution of Wasted Turns')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Middle: Best score achieved at which turn
ax = axes[1]
bins = np.arange(0, 21, 1)
ax.hist(v3_best_pos, bins=bins, alpha=0.6, color=V3_COLOR,
        label=f'v3 (mean={np.mean(v3_best_pos):.1f})', edgecolor='white')
ax.hist(nt_best_pos, bins=bins, alpha=0.6, color=NT_COLOR,
        label=f'NT (mean={np.mean(nt_best_pos):.1f})', edgecolor='white')
ax.set_xlabel('Turn Number')
ax.set_ylabel('Count')
ax.set_title('Turn at Which Best Score Is Achieved')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Right: Summary metrics
ax = axes[2]
metrics = ['Valid Sub\nRate (%)', 'Avg Public\nPercentile', 'Avg\nTurns', 'Wasted\nTurns (%)']

v3_pcts_vals = [r.get("public_percentile") for r in v3_results.values() if r.get("public_percentile") is not None]
nt_pcts_vals = [r.get("public_percentile") for r in nt_results.values() if r.get("public_percentile") is not None]
v3_valid = sum(1 for r in v3_results.values() if r.get("valid_submission"))
nt_valid = sum(1 for r in nt_results.values() if r.get("valid_submission"))

v3_vals = [
    v3_valid / max(len(v3_results), 1) * 100,
    np.mean(v3_pcts_vals) if v3_pcts_vals else 0,
    np.mean([t['num_turns'] for t in v3_tasks]),
    v3_total_wasted / max(v3_total, 1) * 100,
]
nt_vals = [
    nt_valid / max(len(nt_results), 1) * 100,
    np.mean(nt_pcts_vals) if nt_pcts_vals else 0,
    np.mean([t['num_turns'] for t in nt_tasks]),
    nt_total_wasted / max(nt_total, 1) * 100,
]

x = np.arange(len(metrics))
width = 0.35
bars1 = ax.bar(x - width/2, v3_vals, width, label='v3', color=V3_COLOR, alpha=0.8)
bars2 = ax.bar(x + width/2, nt_vals, width, label='No-Terminate', color=NT_COLOR, alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_title('Overall Metrics Comparison')
ax.legend()
for bar_group in [bars1, bars2]:
    for bar in bar_group:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                f'{h:.1f}', ha='center', va='bottom', fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('EET v3 vs No-Terminate: Efficiency Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figNT_E_efficiency.png'), bbox_inches='tight')
plt.close()
print("Saved figNT_E_efficiency.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure NT-F: Iteration Efficiency — Improvement at Checkpoints
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Figure NT-F: Iteration Efficiency ===")

fig, ax = plt.subplots(figsize=(10, 6))

def get_improvement_at_turns(tasks, checkpoints):
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
                last = None
                for s in reversed(best_scores):
                    if s is not None:
                        last = s
                        break
                if last is not None:
                    imp = (baseline - last) / baseline * 100
                    improvements[cp].append(imp)
    return {cp: np.mean(v) if v else 0 for cp, v in improvements.items()}

checkpoints = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18]
v3_imp = get_improvement_at_turns(v3_tasks, checkpoints)
nt_imp = get_improvement_at_turns(nt_tasks, checkpoints)

x_v3 = [cp for cp in checkpoints if cp in v3_imp]
y_v3 = [v3_imp[cp] for cp in x_v3]
x_nt = [cp for cp in checkpoints if cp in nt_imp]
y_nt = [nt_imp[cp] for cp in x_nt]

ax.plot(x_v3, y_v3, color=V3_COLOR, linewidth=2.5, label=V3_LABEL, marker='o', markersize=6)
ax.plot(x_nt, y_nt, color=NT_COLOR, linewidth=2.5, label=NT_LABEL, marker='s', markersize=6)

ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Turn Number')
ax.set_ylabel('Mean Improvement over Baseline (%)')
ax.set_title('Iteration Efficiency: Improvement at Checkpoints')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figNT_F_iteration_efficiency.png'), bbox_inches='tight')
plt.close()
print("Saved figNT_F_iteration_efficiency.png")


# ═══════════════════════════════════════════════════════════════════════
# Print Summary Statistics
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 130)
print("DETAILED SUMMARY: EET v3 vs No-Terminate")
print("=" * 130)

# Per-task table
print(f"\n{'Task':<45} {'v3_pub%':>8} {'NT_pub%':>8} {'diff':>7} {'v3_prv%':>8} {'NT_prv%':>8} {'diff':>7} {'v3_t':>5} {'NT_t':>5}")
print("-" * 130)

for name in all_task_names:
    r3 = v3_results.get(name, {})
    rn = nt_results.get(name, {})
    p3_pub = r3.get("public_percentile")
    pn_pub = rn.get("public_percentile")
    p3_prv = r3.get("private_percentile")
    pn_prv = rn.get("private_percentile")
    t3 = r3.get("total_turns", 0)
    tn = rn.get("total_turns", 0)

    p3s = f"{p3_pub:.1f}" if p3_pub is not None else "N/A"
    pns = f"{pn_pub:.1f}" if pn_pub is not None else "N/A"
    ds_pub = f"{pn_pub-p3_pub:+.1f}" if (p3_pub is not None and pn_pub is not None) else "---"
    p3ps = f"{p3_prv:.1f}" if p3_prv is not None else "N/A"
    pnps = f"{pn_prv:.1f}" if pn_prv is not None else "N/A"
    ds_prv = f"{pn_prv-p3_prv:+.1f}" if (p3_prv is not None and pn_prv is not None) else "---"

    display_name = name.replace("playground-series-", "ps-")
    print(f"  {display_name:<43} {p3s:>8} {pns:>8} {ds_pub:>7} {p3ps:>8} {pnps:>8} {ds_prv:>7} {t3:>5} {tn:>5}")

# Summary stats
print("\n" + "=" * 130)
print(f"  {'Metric':<40} {'EET v3':>15} {'No-Terminate':>15} {'Change':>12}")
print(f"  {'-'*85}")

v3_valid_count = sum(1 for r in v3_results.values() if r.get("valid_submission"))
nt_valid_count = sum(1 for r in nt_results.values() if r.get("valid_submission"))
v3_all_pcts = [r.get("public_percentile") for r in v3_results.values() if r.get("public_percentile") is not None]
nt_all_pcts = [r.get("public_percentile") for r in nt_results.values() if r.get("public_percentile") is not None]
v3_prv_pcts = [r.get("private_percentile") for r in v3_results.values() if r.get("private_percentile") is not None]
nt_prv_pcts = [r.get("private_percentile") for r in nt_results.values() if r.get("private_percentile") is not None]

print(f"  {'Valid submissions':<40} {f'{v3_valid_count}/{len(v3_results)}':>15} {f'{nt_valid_count}/{len(nt_results)}':>15} {f'{nt_valid_count-v3_valid_count:+d}':>12}")
print(f"  {'Valid submission rate':<40} {f'{v3_valid_count/max(len(v3_results),1)*100:.1f}%':>15} {f'{nt_valid_count/max(len(nt_results),1)*100:.1f}%':>15}")
print(f"  {'Avg public percentile':<40} {f'{np.mean(v3_all_pcts):.1f}':>15} {f'{np.mean(nt_all_pcts):.1f}':>15} {f'{np.mean(nt_all_pcts)-np.mean(v3_all_pcts):+.1f}':>12}")
print(f"  {'Median public percentile':<40} {f'{np.median(v3_all_pcts):.1f}':>15} {f'{np.median(nt_all_pcts):.1f}':>15} {f'{np.median(nt_all_pcts)-np.median(v3_all_pcts):+.1f}':>12}")
if v3_prv_pcts and nt_prv_pcts:
    print(f"  {'Avg private percentile':<40} {f'{np.mean(v3_prv_pcts):.1f}':>15} {f'{np.mean(nt_prv_pcts):.1f}':>15} {f'{np.mean(nt_prv_pcts)-np.mean(v3_prv_pcts):+.1f}':>12}")
    print(f"  {'Median private percentile':<40} {f'{np.median(v3_prv_pcts):.1f}':>15} {f'{np.median(nt_prv_pcts):.1f}':>15} {f'{np.median(nt_prv_pcts)-np.median(v3_prv_pcts):+.1f}':>12}")

v3_avg_turns = np.mean([t['num_turns'] for t in v3_tasks])
nt_avg_turns = np.mean([t['num_turns'] for t in nt_tasks])
print(f"  {'Avg turns':<40} {v3_avg_turns:>15.1f} {nt_avg_turns:>15.1f} {f'{nt_avg_turns-v3_avg_turns:+.1f}':>12}")
print(f"  {'Wasted turns (%)':<40} {f'{np.mean(v3_wasted):.1f}%':>15} {f'{np.mean(nt_wasted):.1f}%':>15} {f'{np.mean(nt_wasted)-np.mean(v3_wasted):+.1f}':>12}")
print(f"  {'Best score at turn (avg)':<40} {f'{np.mean(v3_best_pos):.1f}':>15} {f'{np.mean(nt_best_pos):.1f}':>15} {f'{np.mean(nt_best_pos)-np.mean(v3_best_pos):+.1f}':>12}")

# Pairwise comparison
if paired_public:
    diffs_pub = [pn - p3 for p3, pn in paired_public]
    better_pub = sum(1 for d in diffs_pub if d > 0)
    worse_pub = sum(1 for d in diffs_pub if d < 0)
    print(f"\n  Pairwise Public LB (n={len(paired_public)}):")
    print(f"    NT better: {better_pub}, NT worse: {worse_pub}, Same: {len(diffs_pub)-better_pub-worse_pub}")
    print(f"    Avg diff: {np.mean(diffs_pub):+.1f}, Median diff: {np.median(diffs_pub):+.1f}")

if paired_prv_valid:
    diffs_prv = [pn - p3 for p3, pn in paired_prv_valid]
    better_prv = sum(1 for d in diffs_prv if d > 0)
    worse_prv = sum(1 for d in diffs_prv if d < 0)
    print(f"\n  Pairwise Private LB (n={len(paired_prv_valid)}):")
    print(f"    NT better: {better_prv}, NT worse: {worse_prv}, Same: {len(diffs_prv)-better_prv-worse_prv}")
    print(f"    Avg diff: {np.mean(diffs_prv):+.1f}, Median diff: {np.median(diffs_prv):+.1f}")

print(f"\n{'='*130}")
print("All figures saved to:", OUTPUT_DIR)

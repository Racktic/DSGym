"""
Compare EET v3 (no system best_score tracking) vs EET v4 (with system best_score tracking).

Generates:
- Fig A: Combined score curves (v3 EET vs v4 EET, same style as fig5)
- Fig B: Per-task percentile comparison (public + private leaderboard)
- Fig C: Score regression / monotonicity analysis
- Fig D: Wasted turns analysis
- Text summary table with all metrics
"""

import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# ─── Config ───────────────────────────────────────────────────────────
RESULTS_DIR = "/data/fnie/qixin/DSGym/evaluation_results"
EET_V3_DIR = os.path.join(RESULTS_DIR, "eet_qwen3_235b_easy_v3")
EET_V4_DIR = os.path.join(RESULTS_DIR, "eet_qwen3_235b_easy_v4")
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
        out[name] = {
            "public_score": details.get("public_score"),
            "private_score": details.get("private_score"),
            "public_percentile": details.get("public_percentile"),
            "public_medal": details.get("public_medal", "none"),
            "public_rank": details.get("public_rank"),
            "public_above_median": details.get("public_above_median"),
            "valid_submission": kaggle.get("success", False) and not kaggle.get("skipped", False),
            "total_turns": r.get("total_turns", 0),
            "leaderboard_stats": details.get("leaderboard_stats", {}),
        }
    return out


# ─── Score Curve Helpers ──────────────────────────────────────────────

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
    """Get all individual normalized curves (padded)."""
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
v4_tasks = load_trajectories(EET_V4_DIR)
v3_results = load_results(EET_V3_DIR)
v4_results = load_results(EET_V4_DIR)
print(f"EET v3: {len(v3_tasks)} trajectories, {len(v3_results)} results")
print(f"EET v4: {len(v4_tasks)} trajectories, {len(v4_results)} results")


# ═══════════════════════════════════════════════════════════════════════
# Figure A: Combined Score Curves (same style as fig5)
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Figure A: Combined Score Curves ===")

fig, ax = plt.subplots(figsize=(10, 6))

# Get individual curves for shading
v3_curves, v3_len = get_all_curves_padded(v3_tasks, max_len=20)
v4_curves, v4_len = get_all_curves_padded(v4_tasks, max_len=20)

# Plot individual curves (light)
for curve in v3_curves:
    ax.plot(range(len(curve)), curve, color='#55A868', alpha=0.08, linewidth=0.7)
for curve in v4_curves:
    ax.plot(range(len(curve)), curve, color='#E07B39', alpha=0.08, linewidth=0.7)

# Mean curves
v3_mean, v3_mlen = get_mean_curve(v3_tasks, max_len=20)
v4_mean, v4_mlen = get_mean_curve(v4_tasks, max_len=20)

if len(v3_mean) > 0:
    ax.plot(range(v3_mlen), v3_mean, color='#55A868', linewidth=2.5,
            label='EET v3 (no system tracking)', marker='o', markersize=5)
if len(v4_mean) > 0:
    ax.plot(range(v4_mlen), v4_mean, color='#E07B39', linewidth=2.5,
            label='EET v4 (system best_score)', marker='s', markersize=5)

ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Turn Number')
ax.set_ylabel('Mean Improvement over Baseline (%)')
ax.set_title('Score Improvement Comparison: EET v3 vs EET v4 (system best_score tracking)')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figA_eet_v3_v4_score_curves.png'), bbox_inches='tight')
plt.close()
print("Saved figA_eet_v3_v4_score_curves.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure B: Per-task Percentile Comparison (Public Leaderboard)
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Figure B: Percentile Comparison ===")

# Collect paired data
all_tasks = sorted(set(list(v3_results.keys()) + list(v4_results.keys())))
paired_public = []
paired_names = []
v3_only_pct = []
v4_only_pct = []

for name in all_tasks:
    r3 = v3_results.get(name, {})
    r4 = v4_results.get(name, {})
    p3 = r3.get("public_percentile")
    p4 = r4.get("public_percentile")
    if p3 is not None and p4 is not None:
        paired_public.append((p3, p4))
        paired_names.append(name.replace("playground-series-", "ps-"))
    elif p3 is not None:
        v3_only_pct.append((name, p3))
    elif p4 is not None:
        v4_only_pct.append((name, p4))

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: Scatter plot with diagonal
ax = axes[0]
if paired_public:
    x = [p[0] for p in paired_public]
    y = [p[1] for p in paired_public]
    colors_scatter = ['#55A868' if yi > xi else '#C44E52' if yi < xi else '#888'
                      for xi, yi in zip(x, y)]
    ax.scatter(x, y, c=colors_scatter, s=60, alpha=0.8, edgecolors='white', linewidth=0.5, zorder=5)
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Equal performance')
    ax.set_xlabel('EET v3 Percentile (public)')
    ax.set_ylabel('EET v4 Percentile (public)')
    ax.set_title(f'Per-Task Percentile: v3 vs v4 (n={len(paired_public)})')

    # Annotate tasks with large differences
    for i, (xi, yi) in enumerate(zip(x, y)):
        if abs(yi - xi) > 20:
            ax.annotate(paired_names[i], (xi, yi), fontsize=7, alpha=0.7,
                        xytext=(5, 5), textcoords='offset points')

    # Summary stats in text box
    better = sum(1 for xi, yi in zip(x, y) if yi > xi)
    worse = sum(1 for xi, yi in zip(x, y) if yi < xi)
    same = len(x) - better - worse
    avg_diff = np.mean([yi - xi for xi, yi in zip(x, y)])
    textstr = (f'v4 better: {better}\nv4 worse: {worse}\n'
               f'Same: {same}\nAvg diff: {avg_diff:+.1f}')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend(loc='lower right')
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

# Right: Bar chart per task (sorted by diff)
ax = axes[1]
if paired_public:
    diffs = [(paired_names[i], paired_public[i][1] - paired_public[i][0])
             for i in range(len(paired_public))]
    diffs.sort(key=lambda x: x[1])
    names_sorted = [d[0] for d in diffs]
    vals_sorted = [d[1] for d in diffs]
    bar_colors = ['#55A868' if v > 0 else '#C44E52' for v in vals_sorted]
    bars = ax.barh(range(len(names_sorted)), vals_sorted, color=bar_colors, alpha=0.8)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=7)
    ax.set_xlabel('Percentile Change (v4 - v3)')
    ax.set_title('Per-Task Percentile Change (Public LB)')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')

plt.suptitle('EET v3 vs v4: Public Leaderboard Percentile Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figB_eet_v3_v4_percentile.png'), bbox_inches='tight')
plt.close()
print("Saved figB_eet_v3_v4_percentile.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure C: Score Monotonicity / Regression Analysis
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Figure C: Score Monotonicity ===")

def analyze_monotonicity(tasks, label):
    """Check if per-turn scores are monotonically improving."""
    results = {
        'total_tasks': 0,
        'tasks_with_regression': 0,
        'total_regressions': 0,  # total number of regression events
        'regression_magnitudes': [],  # how much score worsened
        'per_turn_regression_count': defaultdict(int),
    }

    for task in tasks:
        if not task['success'] or not task['turns']:
            continue
        results['total_tasks'] += 1

        scores = [t.get('score') for t in task['turns']]
        valid_scores = [(i, s) for i, s in enumerate(scores) if s is not None]
        if len(valid_scores) < 2:
            continue

        # Determine direction
        baseline = task.get('baseline_score')
        final = task.get('final_best_score')
        if baseline is not None and final is not None:
            lower_is_better = final < baseline
        else:
            lower_is_better = True

        has_regression = False
        for j in range(1, len(valid_scores)):
            prev_i, prev_s = valid_scores[j-1]
            curr_i, curr_s = valid_scores[j]
            if lower_is_better:
                if curr_s > prev_s:  # score got worse
                    has_regression = True
                    results['total_regressions'] += 1
                    results['regression_magnitudes'].append(curr_s - prev_s)
                    results['per_turn_regression_count'][curr_i] += 1
            else:
                if curr_s < prev_s:
                    has_regression = True
                    results['total_regressions'] += 1
                    results['regression_magnitudes'].append(prev_s - curr_s)
                    results['per_turn_regression_count'][curr_i] += 1

        if has_regression:
            results['tasks_with_regression'] += 1

    return results

v3_mono = analyze_monotonicity(v3_tasks, "v3")
v4_mono = analyze_monotonicity(v4_tasks, "v4")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Left: Tasks with regression
ax = axes[0]
labels = ['EET v3', 'EET v4']
regression_pct = [
    v3_mono['tasks_with_regression'] / max(v3_mono['total_tasks'], 1) * 100,
    v4_mono['tasks_with_regression'] / max(v4_mono['total_tasks'], 1) * 100,
]
no_regression_pct = [100 - r for r in regression_pct]
bars1 = ax.bar(labels, no_regression_pct, color='#55A868', alpha=0.8, label='Monotonic (no regression)')
bars2 = ax.bar(labels, regression_pct, bottom=no_regression_pct, color='#C44E52', alpha=0.8, label='Has regression')
ax.set_ylabel('% of Tasks')
ax.set_title('Score Monotonicity')
ax.legend()
for i, (nr, r) in enumerate(zip(no_regression_pct, regression_pct)):
    ax.text(i, nr/2, f'{nr:.0f}%', ha='center', va='center', fontweight='bold', color='white')
    if r > 3:
        ax.text(i, nr + r/2, f'{r:.0f}%', ha='center', va='center', fontweight='bold', color='white')
ax.set_ylim(0, 110)

# Middle: Total regression events per turn
ax = axes[1]
max_turn = 20
v3_per_turn = [v3_mono['per_turn_regression_count'].get(i, 0) for i in range(max_turn)]
v4_per_turn = [v4_mono['per_turn_regression_count'].get(i, 0) for i in range(max_turn)]
x = np.arange(max_turn)
width = 0.35
ax.bar(x - width/2, v3_per_turn, width, label='EET v3', color='#55A868', alpha=0.8)
ax.bar(x + width/2, v4_per_turn, width, label='EET v4', color='#E07B39', alpha=0.8)
ax.set_xlabel('Turn Number')
ax.set_ylabel('Regression Events')
ax.set_title('Score Regressions per Turn')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Right: Raw score traces for a sample task (showing regression in v3 vs monotonic in v4)
ax = axes[2]
# Find a task present in both with regression in v3
v3_by_name = {t['task_id']: t for t in v3_tasks}
v4_by_name = {t['task_id']: t for t in v4_tasks}
example_task = None
for task in v3_tasks:
    if not task['success']:
        continue
    scores = [t['score'] for t in task['turns'] if t['score'] is not None]
    if len(scores) >= 3:
        # Check if non-monotonic
        baseline = task.get('baseline_score')
        final = task.get('final_best_score')
        if baseline and final and final < baseline:
            for j in range(1, len(scores)):
                if scores[j] > scores[j-1]:
                    if task['task_id'] in v4_by_name:
                        example_task = task['task_id']
                        break
    if example_task:
        break

if example_task:
    t3 = v3_by_name[example_task]
    t4 = v4_by_name[example_task]
    s3 = [t['score'] for t in t3['turns']]
    s4 = [t['score'] for t in t4['turns']]
    ax.plot(range(len(s3)), s3, 'o-', color='#55A868', label='v3', markersize=4, linewidth=1.5)
    ax.plot(range(len(s4)), s4, 's-', color='#E07B39', label='v4', markersize=4, linewidth=1.5)
    if t3.get('baseline_score'):
        ax.axhline(t3['baseline_score'], color='gray', linestyle=':', alpha=0.5, label='Baseline')
    ax.set_xlabel('Turn')
    ax.set_ylabel('Score')
    ax.set_title(f'Example: {example_task[:35]}...')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'No regression\nexample found', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Example Task')

plt.suptitle('Score Regression Analysis: EET v3 vs v4', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figC_eet_v3_v4_monotonicity.png'), bbox_inches='tight')
plt.close()
print("Saved figC_eet_v3_v4_monotonicity.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure D: Wasted Turns & Efficiency
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Figure D: Wasted Turns & Efficiency ===")

def compute_wasted_turns(tasks):
    """Compute turns after best score (wasted)."""
    wasted_per_task = []
    total_turns_all = 0
    total_wasted = 0
    best_turn_positions = []

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
v4_wasted, v4_total, v4_total_wasted, v4_best_pos = compute_wasted_turns(v4_tasks)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Left: Wasted turns distribution
ax = axes[0]
bins = np.arange(0, 105, 10)
ax.hist(v3_wasted, bins=bins, alpha=0.6, color='#55A868', label=f'v3 (mean={np.mean(v3_wasted):.1f}%)', edgecolor='white')
ax.hist(v4_wasted, bins=bins, alpha=0.6, color='#E07B39', label=f'v4 (mean={np.mean(v4_wasted):.1f}%)', edgecolor='white')
ax.set_xlabel('Wasted Turns (%)')
ax.set_ylabel('Count')
ax.set_title('Distribution of Wasted Turns')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Middle: Best score achieved at which turn
ax = axes[1]
bins = np.arange(0, 21, 1)
ax.hist(v3_best_pos, bins=bins, alpha=0.6, color='#55A868', label=f'v3 (mean={np.mean(v3_best_pos):.1f})', edgecolor='white')
ax.hist(v4_best_pos, bins=bins, alpha=0.6, color='#E07B39', label=f'v4 (mean={np.mean(v4_best_pos):.1f})', edgecolor='white')
ax.set_xlabel('Turn Number')
ax.set_ylabel('Count')
ax.set_title('Turn at Which Best Score Is Achieved')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Right: Summary bar chart
ax = axes[2]
metrics = ['Valid Sub\nRate (%)', 'Avg\nPercentile', 'Avg\nTurns', 'Wasted\nTurns (%)']

# Compute percentiles
v3_pcts = [r.get("public_percentile") for r in v3_results.values() if r.get("public_percentile") is not None]
v4_pcts = [r.get("public_percentile") for r in v4_results.values() if r.get("public_percentile") is not None]
v3_valid = sum(1 for r in v3_results.values() if r.get("valid_submission"))
v4_valid = sum(1 for r in v4_results.values() if r.get("valid_submission"))

v3_vals = [
    v3_valid / max(len(v3_results), 1) * 100,
    np.mean(v3_pcts) if v3_pcts else 0,
    np.mean([t['num_turns'] for t in v3_tasks]),
    v3_total_wasted / max(v3_total, 1) * 100,
]
v4_vals = [
    v4_valid / max(len(v4_results), 1) * 100,
    np.mean(v4_pcts) if v4_pcts else 0,
    np.mean([t['num_turns'] for t in v4_tasks]),
    v4_total_wasted / max(v4_total, 1) * 100,
]

x = np.arange(len(metrics))
width = 0.35
bars1 = ax.bar(x - width/2, v3_vals, width, label='EET v3', color='#55A868', alpha=0.8)
bars2 = ax.bar(x + width/2, v4_vals, width, label='EET v4', color='#E07B39', alpha=0.8)
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

plt.suptitle('Turn Efficiency Analysis: EET v3 vs v4', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figD_eet_v3_v4_efficiency.png'), bbox_inches='tight')
plt.close()
print("Saved figD_eet_v3_v4_efficiency.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure E: Private Leaderboard Comparison (if available)
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Figure E: Private Leaderboard ===")

# Check if private scores exist
v3_has_private = any(
    r.get("leaderboard_stats", {}).get("private_scores")
    for r in v3_results.values()
)
v4_has_private = any(
    r.get("leaderboard_stats", {}).get("private_scores")
    for r in v4_results.values()
)

# Compute private percentile from private_scores + our score
def compute_private_percentile(our_score, lb_stats):
    """Compute percentile on private leaderboard."""
    if not our_score or our_score == '':
        return None
    private_scores = lb_stats.get("private_scores", [])
    if not private_scores:
        return None
    try:
        our_score = float(our_score)
    except (ValueError, TypeError):
        return None
    # Lower score = better rank for most metrics
    # But we don't know direction, so use the public score distribution pattern
    total = len(private_scores)
    # Count how many we beat
    better_count = sum(1 for s in private_scores if s > our_score)
    return better_count / total * 100


# Use public scores to compute percentile directly from raw scores
def compute_percentile_from_scores(our_public_score, lb_stats):
    """Recompute public percentile from raw leaderboard scores."""
    public_scores = lb_stats.get("public_scores", [])
    if not public_scores or our_public_score is None:
        return None
    try:
        our_score = float(our_public_score)
    except (ValueError, TypeError):
        return None
    total = len(public_scores)
    better_count = sum(1 for s in public_scores if s > our_score)
    return better_count / total * 100


# Build comprehensive table
print("\n" + "=" * 120)
print(f"{'Task':<45} {'v3_pub%':>8} {'v4_pub%':>8} {'diff':>7} {'v3_turns':>8} {'v4_turns':>8} {'v3_score':>10} {'v4_score':>10}")
print("=" * 120)

paired_data = []
for name in all_tasks:
    r3 = v3_results.get(name, {})
    r4 = v4_results.get(name, {})
    p3 = r3.get("public_percentile")
    p4 = r4.get("public_percentile")
    s3 = r3.get("public_score")
    s4 = r4.get("public_score")
    t3 = r3.get("total_turns", 0)
    t4 = r4.get("total_turns", 0)

    p3s = f"{p3:.1f}" if p3 is not None else "N/A"
    p4s = f"{p4:.1f}" if p4 is not None else "N/A"
    try:
        s3s = f"{float(s3):.5f}" if s3 is not None and s3 != '' else "N/A"
    except (ValueError, TypeError):
        s3s = "N/A"
    try:
        s4s = f"{float(s4):.5f}" if s4 is not None and s4 != '' else "N/A"
    except (ValueError, TypeError):
        s4s = "N/A"
    ds = f"{p4-p3:+.1f}" if (p3 is not None and p4 is not None) else "---"

    display_name = name.replace("playground-series-", "ps-")
    print(f"  {display_name:<43} {p3s:>8} {p4s:>8} {ds:>7} {t3:>8} {t4:>8} {s3s:>10} {s4s:>10}")

    if p3 is not None and p4 is not None:
        paired_data.append((name, p3, p4, s3, s4))

# Summary
print("=" * 120)
v3_all_pcts = [r.get("public_percentile") for r in v3_results.values() if r.get("public_percentile") is not None]
v4_all_pcts = [r.get("public_percentile") for r in v4_results.values() if r.get("public_percentile") is not None]
v3_valid_count = sum(1 for r in v3_results.values() if r.get("valid_submission"))
v4_valid_count = sum(1 for r in v4_results.values() if r.get("valid_submission"))

print(f"\n  SUMMARY:")
print(f"  {'Metric':<35} {'EET v3':>12} {'EET v4':>12} {'Change':>12}")
print(f"  {'-'*73}")
print(f"  {'Valid submissions':<35} {f'{v3_valid_count}/{len(v3_results)}':>12} {f'{v4_valid_count}/{len(v4_results)}':>12} {f'{v4_valid_count-v3_valid_count:+d}':>12}")
print(f"  {'Valid submission rate':<35} {f'{v3_valid_count/max(len(v3_results),1)*100:.1f}%':>12} {f'{v4_valid_count/max(len(v4_results),1)*100:.1f}%':>12}")
print(f"  {'Avg public percentile':<35} {f'{np.mean(v3_all_pcts):.1f}':>12} {f'{np.mean(v4_all_pcts):.1f}':>12} {f'{np.mean(v4_all_pcts)-np.mean(v3_all_pcts):+.1f}':>12}")
print(f"  {'Median public percentile':<35} {f'{np.median(v3_all_pcts):.1f}':>12} {f'{np.median(v4_all_pcts):.1f}':>12} {f'{np.median(v4_all_pcts)-np.median(v3_all_pcts):+.1f}':>12}")

if paired_data:
    diffs_p = [p4-p3 for _, p3, p4, _, _ in paired_data]
    better = sum(1 for d in diffs_p if d > 0)
    worse = sum(1 for d in diffs_p if d < 0)
    same = len(diffs_p) - better - worse
    print(f"  {'Pairwise: v4 better':<35} {better:>12}")
    print(f"  {'Pairwise: v4 worse':<35} {worse:>12}")
    print(f"  {'Pairwise: same':<35} {same:>12}")
    print(f"  {'Pairwise: avg diff':<35} {'':>12} {'':>12} {f'{np.mean(diffs_p):+.1f}':>12}")

v3_avg_turns = np.mean([t['num_turns'] for t in v3_tasks])
v4_avg_turns = np.mean([t['num_turns'] for t in v4_tasks])
print(f"  {'Avg turns':<35} {v3_avg_turns:>12.1f} {v4_avg_turns:>12.1f}")
v3_reg = f"{v3_mono['tasks_with_regression']}/{v3_mono['total_tasks']}"
v4_reg = f"{v4_mono['tasks_with_regression']}/{v4_mono['total_tasks']}"
print(f"  {'Score regressions (tasks)':<35} {v3_reg:>12} {v4_reg:>12}")
print(f"  {'Score regressions (events)':<35} {v3_mono['total_regressions']:>12} {v4_mono['total_regressions']:>12}")
v3_waste = f"{v3_total_wasted}/{v3_total} ({v3_total_wasted/max(v3_total,1)*100:.1f}%)"
v4_waste = f"{v4_total_wasted}/{v4_total} ({v4_total_wasted/max(v4_total,1)*100:.1f}%)"
print(f"  {'Wasted turns':<35} {v3_waste:>24} {v4_waste:>24}")

print("\nDone! All figures saved to", OUTPUT_DIR)

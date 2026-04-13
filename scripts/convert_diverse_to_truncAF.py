"""
Convert diverse teacher trajectories (Claude/GPT/Gemini) to truncAF SFT format.

Input: trajectory directory
Output: truncAF JSON file

Steps:
1. Find best score turn from task_memory
2. Truncate conversation to that turn
3. Find exec output of best turn (user message AFTER best turn assistant in full conv)
4. Append: user (exec output + FINAL SUBMISSION prompt) + assistant (best turn code repeated)
5. Clean: remove memory sections, [Step X/Y], convert assistant to V6 (<goal> + <python> only)
6. Merge consecutive user messages
7. Filter: every assistant must have <goal> + <python>
"""

import json
import re
import os
import glob
import argparse


def remove_memory_sections(text):
    """Remove task memory, cross-task memory, reference approach, step markers."""
    text = re.sub(r'=== TASK MEMORY \(Previous Attempts\) ===.*?=== END TASK MEMORY ===\n*', '', text, flags=re.DOTALL)
    text = re.sub(r'=== CROSS-TASK EXPERIENCE MEMORY ===.*?=== END CROSS-TASK MEMORY ===\n*', '', text, flags=re.DOTALL)
    text = re.sub(r'--- Reference approach \(.*?\) ---.*?--- End reference ---\n*', '', text, flags=re.DOTALL)
    text = text.replace('No previous attempts yet.\n\n', '')
    text = text.replace('No previous attempts yet.', '')
    text = re.sub(r'Model usage frequency across tasks:.*?Consider exploring under-represented approaches for diversity\.\n*', '', text, flags=re.DOTALL)
    text = re.sub(r'\[Step \d+/\d+\]\n?', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def convert_assistant_to_v6(content):
    """Extract <goal> + <python> only. Do NOT fix unclosed tags (truncated code)."""
    goal_match = re.search(r'<goal>(.*?)</goal>', content, re.DOTALL)
    python_match = re.search(r'<python>(.*?)</python>', content, re.DOTALL)

    # Try <reasoning> as fallback for <goal>
    if not goal_match:
        goal_match = re.search(r'<reasoning>(.*?)</reasoning>', content, re.DOTALL)
        if not goal_match and '<reasoning>' in content:
            # Unclosed <reasoning>
            goal_match = re.search(r'<reasoning>(.*)', content, re.DOTALL)

    if not goal_match:
        return content

    result = f'<goal>{goal_match.group(1).strip()}</goal>'
    if python_match:
        result += f'\n\n<python>\n{python_match.group(1).strip()}\n</python>'

    # Remove any <answer> that might be left
    result = re.sub(r'\s*<answer>.*?</answer>\s*', '', result, flags=re.DOTALL)
    if '<answer>' in result:
        result = result[:result.index('<answer>')].strip()

    return result


def merge_consecutive_users(messages):
    """Merge consecutive user messages into one."""
    merged = []
    for msg in messages:
        if merged and merged[-1]['role'] == 'user' and msg['role'] == 'user':
            merged[-1]['content'] = merged[-1]['content'] + '\n\n' + msg['content']
        else:
            merged.append({'role': msg['role'], 'content': msg['content']})
    return merged


def compress_exec_output(text, max_warn_chars=1800):
    """Truncate <information> content to max_warn_chars if it contains warnings/noise."""
    def truncate_info(match):
        inner = match.group(1)
        has_warning = bool(re.search(
            r'\[Warning\]|DeprecationWarning|FutureWarning|UserWarning|ConvergenceWarning',
            inner
        ))
        if has_warning and len(inner) > max_warn_chars:
            return f'<information>{inner[:max_warn_chars]}\n... (truncated)</information>'
        return f'<information>{inner}</information>'

    text = re.sub(r'<information>(.*?)</information>', truncate_info, text, flags=re.DOTALL)
    return text


def build_truncAF(traj_path, split_name):
    """Build truncAF from a single trajectory file."""
    with open(traj_path) as f:
        traj = json.load(f)

    if not traj.get('success', False):
        return None

    conv = traj.get('conversation', [])
    task_memory = traj.get('task_memory', [])
    if not conv or not task_memory:
        return None

    final_best = traj.get('final_best_score')
    if final_best is None:
        return None

    # Step 1: Find best turn (score matches final_best_score)
    best_turn_idx = None
    for i, tm in enumerate(task_memory):
        if tm.get('score') is not None and abs(tm['score'] - final_best) < 1e-9:
            best_turn_idx = i
            break

    if best_turn_idx is None:
        return None

    # Skip trajectories where best score was achieved too early (turn 0 or 1)
    # These have too little learning signal for SFT
    if best_turn_idx <= 1:
        return None

    # Step 2: Truncate conversation to best turn
    target_asst_count = best_turn_idx + 1
    truncated = []
    asst_count = 0
    for m in conv:
        truncated.append(m)
        if m['role'] == 'assistant':
            asst_count += 1
            if asst_count >= target_asst_count:
                break
    while truncated and truncated[-1]['role'] != 'assistant':
        truncated.pop()
    if not truncated:
        return None

    # Best turn's assistant content (for fallback)
    best_turn_asst = convert_assistant_to_v6(truncated[-1]['content'])

    # Step 3: Find exec output of best turn
    # = the user message right AFTER the best turn assistant in the FULL conversation
    asst_count = 0
    exec_output = ''
    for i, m in enumerate(conv):
        if m['role'] == 'assistant':
            asst_count += 1
        elif m['role'] == 'user' and asst_count == target_asst_count:
            info = re.search(r'(<information>.*?</information>)', m['content'], re.DOTALL)
            if info:
                exec_output = compress_exec_output(info.group(1))
            break

    # Step 4: Build final user + assistant
    final_user = exec_output + '\n\n[ACTION: FINAL SUBMISSION]\n\nGenerate your final submission now using your best approach. Save predictions to /submission/submission.csv.'

    # For final assistant: use the LAST assistant from original trajectory
    # If it has no <python> (e.g., only <answer>), skip this trajectory entirely
    # (no fallback to best turn — that would teach model to submit validation code)
    final_asst = None
    for m in reversed(conv):
        if m['role'] == 'assistant':
            final_asst = convert_assistant_to_v6(m['content'])
            # Remove <answer>
            final_asst = re.sub(r'\s*<answer>.*?</answer>\s*', '', final_asst, flags=re.DOTALL)
            if '<answer>' in final_asst:
                final_asst = final_asst[:final_asst.index('<answer>')].strip()
            break
    if not final_asst or '<python>' not in final_asst:
        return None

    # Step 5: Clean all messages
    cleaned = []
    for m in truncated:
        if m['role'] == 'user':
            cleaned_content = remove_memory_sections(m['content'])
            cleaned_content = compress_exec_output(cleaned_content)
            if cleaned_content:
                cleaned.append({'role': 'user', 'content': cleaned_content})
        elif m['role'] == 'assistant':
            cleaned.append({'role': 'assistant', 'content': convert_assistant_to_v6(m['content'])})
        else:
            cleaned.append({'role': m['role'], 'content': m['content']})

    # Append final submission turn
    cleaned.append({'role': 'user', 'content': final_user})
    cleaned.append({'role': 'assistant', 'content': final_asst})

    # Step 6: Remove bad assistant turns (missing <python>) and their surrounding users
    # A full turn = user(instruction) + assistant(code) + user(exec output)
    # If assistant is bad, remove all three
    bad_indices = set()
    for i, m in enumerate(cleaned):
        if m['role'] == 'assistant' and ('<goal>' not in m['content'] or '<python>' not in m['content']):
            bad_indices.add(i)  # bad assistant
            # Previous user = instruction for this turn
            if i - 1 >= 0 and cleaned[i - 1]['role'] == 'user':
                bad_indices.add(i - 1)
            # Next user = exec output of this bad code
            if i + 1 < len(cleaned) and cleaned[i + 1]['role'] == 'user':
                bad_indices.add(i + 1)
    cleaned = [m for i, m in enumerate(cleaned) if i not in bad_indices]

    # Step 7: Merge consecutive users
    merged = merge_consecutive_users(cleaned)

    if not merged or merged[0]['role'] != 'system':
        return None

    # Validate - every assistant must have <goal> + <python>
    for m in merged:
        if m['role'] == 'assistant':
            if '<goal>' not in m['content'] or '<python>' not in m['content']:
                return None

    meta = {
        'challenge_name': traj.get('challenge_name', ''),
        'split': split_name,
        'teacher': traj.get('model', ''),
        'final_best_score': final_best,
        'baseline_score': traj.get('baseline_score'),
        'best_turn_idx': best_turn_idx,
        'num_turns_truncated': target_asst_count,
        'num_turns_original': len(task_memory),
        'turns': task_memory,
    }

    return {'messages': merged, 'meta': meta}


def main():
    parser = argparse.ArgumentParser(description='Convert trajectories to truncAF SFT format')
    parser.add_argument('--input-dir', required=True, help='Directory with trajectory JSON files')
    parser.add_argument('--split', required=True, help='Split name (easy/hard/swap/mledojo)')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    args = parser.parse_args()

    trajs = sorted(glob.glob(os.path.join(args.input_dir, '*_trajectory.json')))
    results = []
    skipped = 0

    for tf in trajs:
        result = build_truncAF(tf, args.split)
        if result:
            results.append(result)
        else:
            skipped += 1

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f'Input: {len(trajs)} trajectories from {args.input_dir}')
    print(f'Output: {len(results)} samples, {skipped} skipped')
    print(f'Saved to: {args.output}')

    # Quick validation
    total_asst = 0
    bad = 0
    for s in results:
        for m in s['messages']:
            if m['role'] == 'assistant':
                total_asst += 1
                c = m['content']
                if '<goal>' not in c or '<python>' not in c:
                    bad += 1
                if '<search_state>' in c or '<step>' in c or '<answer>' in c or '<reasoning>' in c:
                    bad += 1
    print(f'Validation: {total_asst} assistant msgs, {bad} bad')


if __name__ == '__main__':
    main()

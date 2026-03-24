#!/usr/bin/env python3
"""
Rollout Monitor — watches for new global_step_* directories and analyzes
reward=0 trajectories for error classification.

Monitors for 60 minutes, checking every 90 seconds.
"""

import json
import glob
import os
import sys
import time
import subprocess
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timedelta

# ── Config ──────────────────────────────────────────────────────────────
ROLLOUT_DIR = "/data/fnie/qixin/DSGym/SkyRL/checkpoints/skyrl-dspredict-react-4b/rollout_logs/train/"
REMOTE_HOST = "research-common-30"
REMOTE_PID = 660705
POLL_INTERVAL = 90        # seconds
MONITOR_DURATION = 60 * 60  # 60 minutes in seconds
EXPECTED_FILES_PER_STEP = 8

# ── State ───────────────────────────────────────────────────────────────
seen_steps = set()
step_history = []  # list of dicts: {step, avg_reward, max_reward, failures, error_counts}


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def classify_error(data):
    """Classify the error type for a reward=0 trajectory."""
    # Check reward_info first
    reward_info = data.get("reward_info", "")
    if isinstance(reward_info, dict):
        reward_info = json.dumps(reward_info)
    reward_info_lower = str(reward_info).lower()

    # Check stop_reason
    stop_reason = data.get("stop_reason", "")

    # Collect all env_observations into one string for keyword search
    turns = data.get("turns", [])
    all_obs = ""
    for turn in turns:
        obs = turn.get("env_observation", "")
        if obs:
            all_obs += str(obs) + "\n"
    all_text = (all_obs + str(reward_info)).lower()

    # Classification priority
    if "400 bad request" in all_text or "400 bad" in all_text:
        return "400_bad_request"
    if "no submission" in reward_info_lower:
        return "no_submission"
    if "kaggle" in reward_info_lower and ("error" in reward_info_lower or "fail" in reward_info_lower):
        return "kaggle_error"
    if "timeouterror" in all_text or "timed out" in all_text or "timeout" in all_text:
        return "timeout"
    if stop_reason == "length":
        return "length_limit"
    return "other"


def analyze_step(step_dir):
    """Analyze all JSON files in a step directory."""
    files = sorted(glob.glob(os.path.join(step_dir, "*.json")))
    if len(files) < EXPECTED_FILES_PER_STEP:
        return None  # step not complete yet

    rewards = []
    error_counts = defaultdict(int)
    failures = 0
    error_details = []

    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, IOError) as e:
            log(f"  WARNING: could not read {f}: {e}")
            continue

        reward = data.get("final_reward", 0)
        rewards.append(reward)

        if reward == 0:
            failures += 1
            err_type = classify_error(data)
            error_counts[err_type] += 1
            error_details.append((os.path.basename(f), err_type))

    if not rewards:
        return None

    avg_reward = sum(rewards) / len(rewards)
    max_reward = max(rewards)

    return {
        "avg_reward": avg_reward,
        "max_reward": max_reward,
        "failures": failures,
        "total": len(rewards),
        "error_counts": dict(error_counts),
        "error_details": error_details,
    }


def check_systemic_issues():
    """Check for systemic problems that warrant killing the training."""
    if len(step_history) < 1:
        return None

    latest = step_history[-1]

    # Rule 1: 400 Bad Request >= 3 in ANY step
    if latest["error_counts"].get("400_bad_request", 0) >= 3:
        return f"400 Bad Request appeared {latest['error_counts']['400_bad_request']} times in step (>=3 threshold)"

    # Rule 2: 3 consecutive steps with avg reward = 0
    if len(step_history) >= 3:
        last3 = step_history[-3:]
        if all(s["avg_reward"] == 0 for s in last3):
            steps_str = ", ".join(s["step"] for s in last3)
            return f"3 consecutive steps with avg reward=0: {steps_str}"

    # Rule 3: Trend — same error type >70% for 3 consecutive steps
    if len(step_history) >= 3:
        last3 = step_history[-3:]
        all_types = set()
        for s in last3:
            all_types.update(s["error_counts"].keys())
        for err_type in all_types:
            dominant = True
            for s in last3:
                total_failures = s["failures"]
                if total_failures == 0:
                    dominant = False
                    break
                ratio = s["error_counts"].get(err_type, 0) / total_failures
                if ratio <= 0.7:
                    dominant = False
                    break
            if dominant:
                return f"Systemic: '{err_type}' >70% of failures for 3 consecutive steps"

    return None


def kill_training(reason):
    """Kill the remote training process."""
    log(f"KILLING TRAINING — reason: {reason}")
    try:
        result = subprocess.run(
            ["ssh", REMOTE_HOST, f"kill {REMOTE_PID}"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            log("Training process killed successfully.")
        else:
            log(f"Kill command returned {result.returncode}: {result.stderr.strip()}")
    except Exception as e:
        log(f"Failed to kill training: {e}")


def get_existing_steps():
    """Get the set of step directory names currently present."""
    if not os.path.isdir(ROLLOUT_DIR):
        return set()
    return {
        d for d in os.listdir(ROLLOUT_DIR)
        if d.startswith("global_step_") and os.path.isdir(os.path.join(ROLLOUT_DIR, d))
    }


def step_sort_key(name):
    """Extract numeric step for sorting."""
    try:
        return int(name.replace("global_step_", ""))
    except ValueError:
        return 0


def main():
    global seen_steps

    log("=" * 70)
    log("Rollout Monitor started")
    log(f"  Directory : {ROLLOUT_DIR}")
    log(f"  Remote    : {REMOTE_HOST} PID {REMOTE_PID}")
    log(f"  Duration  : {MONITOR_DURATION // 60} minutes")
    log(f"  Interval  : {POLL_INTERVAL} seconds")
    log("=" * 70)

    # Initialize with existing steps
    seen_steps = get_existing_steps()
    if seen_steps:
        log(f"Pre-existing steps (will skip): {sorted(seen_steps, key=step_sort_key)}")
    else:
        log("No pre-existing steps found. Waiting for directory/data...")

    end_time = datetime.now() + timedelta(seconds=MONITOR_DURATION)
    check_count = 0

    while datetime.now() < end_time:
        check_count += 1
        current_steps = get_existing_steps()
        new_steps = current_steps - seen_steps

        if not os.path.isdir(ROLLOUT_DIR):
            if check_count % 4 == 1:  # don't spam, report every ~6 min
                log(f"Directory not yet created. Waiting... (check #{check_count})")
            time.sleep(POLL_INTERVAL)
            continue

        if not new_steps:
            remaining = int((end_time - datetime.now()).total_seconds() / 60)
            if check_count % 4 == 1:
                log(f"No new steps. {remaining}min remaining. (check #{check_count})")
            time.sleep(POLL_INTERVAL)
            continue

        # Process new steps in order
        for step_name in sorted(new_steps, key=step_sort_key):
            step_dir = os.path.join(ROLLOUT_DIR, step_name)
            result = analyze_step(step_dir)

            if result is None:
                # Not enough files yet — don't mark as seen, retry later
                log(f"[{step_name}] incomplete (<{EXPECTED_FILES_PER_STEP} files), will retry")
                continue

            seen_steps.add(step_name)

            # Format error breakdown
            err_parts = []
            for etype in ["length_limit", "timeout", "400_bad_request",
                          "no_submission", "kaggle_error", "other"]:
                count = result["error_counts"].get(etype, 0)
                if count > 0:
                    err_parts.append(f"{etype}={count}")
            err_str = ", ".join(err_parts) if err_parts else "none"

            step_num = step_name.replace("global_step_", "")
            log(
                f"[Step {step_num}] reward: avg={result['avg_reward']:.2f}, "
                f"max={result['max_reward']:.2f} | "
                f"failures: {result['failures']}/{result['total']} | "
                f"errors: {err_str}"
            )

            # Store in history
            result["step"] = step_name
            step_history.append(result)

            # Check systemic issues
            issue = check_systemic_issues()
            if issue:
                log(f"SYSTEMIC ISSUE DETECTED: {issue}")
                kill_training(issue)
                log("Monitor exiting after kill.")
                return

        time.sleep(POLL_INTERVAL)

    log("=" * 70)
    log("Monitor duration expired (60 minutes). Summary:")
    if step_history:
        avg_all = sum(s["avg_reward"] for s in step_history) / len(step_history)
        log(f"  Steps analyzed: {len(step_history)}")
        log(f"  Overall avg reward: {avg_all:.3f}")
        total_errors = defaultdict(int)
        for s in step_history:
            for k, v in s["error_counts"].items():
                total_errors[k] += v
        log(f"  Total error breakdown: {dict(total_errors)}")
    else:
        log("  No steps were analyzed during monitoring period.")
    log("=" * 70)


if __name__ == "__main__":
    main()

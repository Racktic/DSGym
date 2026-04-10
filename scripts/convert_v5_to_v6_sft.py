"""Convert V5 SFT data to V6 format.

V5 assistant output:
  <search_state>
    <step>N</step>
    <best_score>X</best_score>
    <baseline_score>Y</baseline_score>
    <goal>...</goal>
  </search_state>
  <python>...</python>

V6 assistant output:
  <goal>...</goal>
  <python>...</python>

Also replaces the system prompt with V6 version.
"""

import json
import re
import os

AIDE_SYSTEM_PROMPT_V6 = """\
You are a Kaggle grandmaster attending a competition.
In order to win this competition, you need to come up with excellent and creative solutions \
and implement them in Python.

At each step, you must output TWO blocks in this exact order:

1. **<goal>** — A brief description (3-5 sentences) of what you plan to do in this step.

2. **<python>** — Your executable Python code implementing the solution.

=== OUTPUT FORMAT ===

<goal>[what you plan to do]</goal>

<python>
[Your executable Python code]
</python>

=== CRITICAL RULES ===

- Always PRINT validation/CV scores when training models:
    print(f"Validation Score: {score:.6f}")
    print(f"CV Score (mean): {mean:.6f}")
- Do NOT use plotting libraries. Use text-based summaries and statistics only.
- Code execution is continuous — variables persist across steps.
- When generating final submission, save to /submission/submission.csv.
"""


def convert_assistant_message(content: str) -> str:
    """Strip <search_state> wrapper, keep only <goal> and <python>."""
    # Extract <goal>...</goal> from inside <search_state>
    goal_match = re.search(r"<goal>(.*?)</goal>", content, re.DOTALL)
    # Extract <python>...</python> (outside search_state)
    python_match = re.search(r"<python>(.*?)</python>", content, re.DOTALL)
    # Extract <answer>...</answer> if present
    answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)

    if not goal_match:
        # No goal tag, return as-is
        return content

    result = f"<goal>{goal_match.group(1).strip()}</goal>"

    if python_match:
        result += f"\n\n<python>\n{python_match.group(1).strip()}\n</python>"

    if answer_match:
        result += f"\n\n<answer>{answer_match.group(1).strip()}</answer>"

    return result


def convert_user_message(content: str) -> str:
    """Update user message: replace references to <search_state><goal> with <goal>."""
    content = content.replace("inside `<search_state><goal>`", "inside `<goal>`")
    content = content.replace("<search_state><goal>", "<goal>")
    return content


def convert_dataset(input_path: str, output_path: str):
    with open(input_path) as f:
        data = json.load(f)

    converted = []
    for sample in data:
        new_messages = []
        for msg in sample["messages"]:
            new_msg = dict(msg)
            if msg["role"] == "system":
                new_msg["content"] = AIDE_SYSTEM_PROMPT_V6
            elif msg["role"] == "assistant":
                new_msg["content"] = convert_assistant_message(msg["content"])
            elif msg["role"] == "user":
                new_msg["content"] = convert_user_message(msg["content"])
            new_messages.append(new_msg)

        converted.append({"messages": new_messages, "meta": sample.get("meta", {})})

    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(converted)} samples: {input_path} -> {output_path}")


if __name__ == "__main__":
    sft_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sft")

    # Convert all truncA and truncB files
    for fname in sorted(os.listdir(sft_dir)):
        if "v5" in fname and ("truncA" in fname or "truncB" in fname) and fname.endswith(".json"):
            input_path = os.path.join(sft_dir, fname)
            output_name = fname.replace("v5", "v6")
            output_path = os.path.join(sft_dir, output_name)
            convert_dataset(input_path, output_path)

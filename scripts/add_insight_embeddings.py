"""
Add insight embeddings to the enriched cross-task memory.

Loads data/memory/cross_task_memory_teacher_v5_enriched.json,
builds "Goal: {plan}\\n\\nInsight: {insight}" for each entry,
embeds via OpenAI text-embedding-3-small (direct API, not LiteLLM),
writes a parallel .npy file.

Output: data/memory/cross_task_memory_teacher_v5_insight_embeddings.npy
        (shape (N, 1536), aligned with enriched.json order)

Usage:
  export OPENAI_API_KEY=...
  .venv/bin/python3 scripts/add_insight_embeddings.py \
    --input data/memory/cross_task_memory_teacher_v5_enriched.json \
    --output-emb data/memory/cross_task_memory_teacher_v5_insight_embeddings.npy
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import numpy as np

try:
    from openai import OpenAI
except ImportError:
    print("Need: pip install openai")
    sys.exit(1)

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None


def _truncate_tokens(text: str, max_tokens: int = 8000) -> str:
    if not text:
        return ""
    if _ENC is None:
        return text[:25000]
    toks = _ENC.encode(text)
    if len(toks) <= max_tokens:
        return text
    return _ENC.decode(toks[:max_tokens])


def build_insight_text(entry: dict) -> str:
    plan = (entry.get("plan") or "").strip()
    insight = (entry.get("insight") or "").strip()
    return f"Goal: {plan}\n\nInsight: {insight}"


def call_embedding(
    client: OpenAI, model: str, text: str, dim: int = 1536, max_retries: int = 3
) -> Optional[List[float]]:
    t = _truncate_tokens(text or "", max_tokens=8000)
    if not t:
        return [0.0] * dim
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=model, input=t)
            return resp.data[0].embedding
        except Exception as e:
            wait = 3 * (attempt + 1)
            print(f"  [embed retry {attempt+1}/{max_retries}] {e!s}; sleep {wait}s", flush=True)
            time.sleep(wait)
    return None


def process(idx: int, entry: dict, client: OpenAI, model: str, dim: int):
    text = build_insight_text(entry)
    emb = call_embedding(client, model, text, dim)
    return idx, emb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-emb", required=True)
    p.add_argument("--model", default="text-embedding-3-small")
    p.add_argument("--emb-dim", type=int, default=1536)
    p.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--max-workers", type=int, default=16)
    args = p.parse_args()

    if not args.openai_api_key:
        print("ERROR: $OPENAI_API_KEY required")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        entries = json.load(f)
    n = len(entries)
    print(f"Loaded {n} entries from {args.input}", flush=True)

    client = OpenAI(api_key=args.openai_api_key)
    emb = np.zeros((n, args.emb_dim), dtype=np.float32)

    t0 = time.time()
    n_done = 0
    n_err = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {
            ex.submit(process, i, entries[i], client, args.model, args.emb_dim): i
            for i in range(n)
        }
        for fut in as_completed(futures):
            idx, result = fut.result()
            if result is None:
                n_err += 1
            else:
                v = np.asarray(result, dtype=np.float32)
                if v.shape[0] != args.emb_dim:
                    # pad or trim
                    padded = np.zeros(args.emb_dim, dtype=np.float32)
                    k = min(v.shape[0], args.emb_dim)
                    padded[:k] = v[:k]
                    v = padded
                emb[idx] = v
            n_done += 1
            if n_done % 50 == 0 or n_done == n:
                elapsed = time.time() - t0
                eta = elapsed / n_done * (n - n_done) if n_done else 0
                print(
                    f"[{n_done}/{n}] elapsed {elapsed:.0f}s eta {eta:.0f}s errors={n_err}",
                    flush=True,
                )

    os.makedirs(os.path.dirname(args.output_emb) or ".", exist_ok=True)
    tmp = args.output_emb + ".tmp.npy"
    np.save(tmp, emb)
    os.replace(tmp, args.output_emb)

    # Sanity check
    nonzero = int((np.linalg.norm(emb, axis=1) > 1e-6).sum())
    print(f"\nDone. Wrote {n} embeddings to {args.output_emb}")
    print(f"Shape: {emb.shape} dtype={emb.dtype}")
    print(f"Non-zero rows: {nonzero}/{n}  Errors: {n_err}  Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

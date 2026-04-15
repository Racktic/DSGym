"""
Enrich cross-task memory entries with structured metadata + task embeddings.

For each memory entry in the input JSON:
  1. Call Claude (claude-sonnet-4.6) to extract {domain, task_type, metric, data_size}
     from the task_description.
  2. Call text-embedding-3-small to embed task_description[:3000] -> 1536-dim vector.

Progressively saves enriched JSON + embeddings .npy so restarts don't lose progress.
Uses ThreadPoolExecutor for parallel API calls.

Usage:
  export LITELLM_API_KEY=xxx
  python scripts/enrich_memory_metadata.py \
      --input data/memory/cross_task_memory_teacher_v5.json \
      --output-json data/memory/cross_task_memory_teacher_v5_enriched.json \
      --output-emb data/memory/cross_task_memory_teacher_v5_embeddings.npy \
      --base-url https://litellm.nbdevenv.xiaoaojianghu.fun \
      --max-workers 16 \
      [--limit 10]
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    print("Need: pip install openai")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Need: pip install numpy")
    sys.exit(1)


# -----------------------------
# Metadata extraction
# -----------------------------

METADATA_SYSTEM_PROMPT = """You classify Kaggle-style ML task descriptions into a controlled taxonomy. Output ONLY strict JSON, no prose, no markdown."""


METADATA_USER_TEMPLATE = """Classify this ML task into a strict JSON object with these four fields. Each field MUST use one of the listed enum values; use "other" / "unknown" when nothing fits.

{{
  "domain": one of ["tabular", "image", "text", "time_series", "graph", "audio", "multimodal", "other"],
  "task_type": one of ["binary_classification", "multiclass_classification", "multilabel_classification", "regression", "ranking", "segmentation", "detection", "generation", "other"],
  "metric": one of ["rmse", "rmsle", "mae", "auc", "log_loss", "f1", "accuracy", "ndcg", "map", "other"],
  "data_size": one of ["small", "medium", "large", "unknown"]  // small <10K rows, medium 10K-1M, large >1M
}}

Rules:
- Output EXACTLY one JSON object. No markdown fences, no comments.
- Use lowercase values exactly as listed above.
- If the task mixes modalities (e.g., image + tabular), use "multimodal".
- For data_size, base the decision on the dataset row count if stated; if unclear, use "unknown".

Task description:
---
{task_description}
---

JSON:"""


ALLOWED_DOMAIN = {"tabular", "image", "text", "time_series", "graph", "audio", "multimodal", "other"}
ALLOWED_TASK_TYPE = {
    "binary_classification", "multiclass_classification", "multilabel_classification",
    "regression", "ranking", "segmentation", "detection", "generation", "other",
}
ALLOWED_METRIC = {"rmse", "rmsle", "mae", "auc", "log_loss", "f1", "accuracy", "ndcg", "map", "other"}
ALLOWED_DATA_SIZE = {"small", "medium", "large", "unknown"}


DEFAULT_METADATA = {
    "domain": "other",
    "task_type": "other",
    "metric": "other",
    "data_size": "unknown",
}


def _parse_metadata_json(text: str) -> Dict[str, str]:
    """Parse LLM output, validate against allowed vocab, fill defaults on failure."""
    if not text:
        return dict(DEFAULT_METADATA)
    # Strip possible code fences
    t = text.strip()
    if t.startswith("```"):
        # Drop fence line
        lines = t.split("\n")
        # Drop first fence and potential lang tag
        lines = lines[1:]
        # Drop trailing fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    # Some models might prefix with "JSON:" etc. — try to locate first '{'
    i = t.find("{")
    j = t.rfind("}")
    if i == -1 or j == -1 or j < i:
        return dict(DEFAULT_METADATA)
    try:
        obj = json.loads(t[i : j + 1])
    except Exception:
        return dict(DEFAULT_METADATA)

    out = dict(DEFAULT_METADATA)
    d = obj.get("domain", "other")
    if isinstance(d, str) and d.lower() in ALLOWED_DOMAIN:
        out["domain"] = d.lower()
    tt = obj.get("task_type", "other")
    if isinstance(tt, str) and tt.lower() in ALLOWED_TASK_TYPE:
        out["task_type"] = tt.lower()
    m = obj.get("metric", "other")
    if isinstance(m, str) and m.lower() in ALLOWED_METRIC:
        out["metric"] = m.lower()
    ds = obj.get("data_size", "unknown")
    if isinstance(ds, str) and ds.lower() in ALLOWED_DATA_SIZE:
        out["data_size"] = ds.lower()
    return out


def call_metadata_llm(
    client: OpenAI,
    model: str,
    task_description: str,
    max_retries: int = 3,
) -> Dict[str, str]:
    """Extract structured metadata via Claude. Defaults on failure."""
    # Full task_description (the interesting parts — data fields, metric, submission format —
    # are NOT at the beginning of the text, so we do not truncate).
    prompt = METADATA_USER_TEMPLATE.format(task_description=task_description or "")
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": METADATA_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            text = resp.choices[0].message.content
            return _parse_metadata_json(text)
        except Exception as e:
            wait = 3 * (attempt + 1)
            print(f"  [metadata retry {attempt+1}/{max_retries}] {e!s}; sleep {wait}s", flush=True)
            time.sleep(wait)
    return dict(DEFAULT_METADATA)


# -----------------------------
# Embeddings
# -----------------------------

def call_embedding(
    client: OpenAI,
    model: str,
    text: str,
    max_retries: int = 3,
    dim: int = 1536,
) -> Optional[List[float]]:
    """Embed text. Returns None on total failure.

    text-embedding-3-small max input is 8191 tokens. A small number of task
    descriptions (e.g., ones with huge column lists) exceed this. Use tiktoken
    for precise truncation — drops the tail (usually generic instruction
    boilerplate) while keeping the informative main body.
    """
    t = text or ""
    if not t:
        # Fill zeros — keeps indices aligned
        return [0.0] * dim
    # Token-aware truncation to 8000 tokens (safety margin under 8191 hard limit)
    try:
        import tiktoken
        _enc = tiktoken.get_encoding("cl100k_base")
        tokens = _enc.encode(t)
        if len(tokens) > 8000:
            t = _enc.decode(tokens[:8000])
    except Exception:
        # Fallback to char-based truncation if tiktoken unavailable
        if len(t) > 25000:
            t = t[:25000]
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=model, input=t)
            return resp.data[0].embedding
        except Exception as e:
            wait = 3 * (attempt + 1)
            print(f"  [embedding retry {attempt+1}/{max_retries}] {e!s}; sleep {wait}s", flush=True)
            time.sleep(wait)
    return None


# -----------------------------
# Worker
# -----------------------------

def process_one(
    idx: int,
    entry: Dict[str, Any],
    metadata_model: str,
    embedding_model: str,
    base_url: str,
    litellm_api_key: str,
    openai_api_key: str,
) -> Tuple[int, Dict[str, str], Optional[List[float]]]:
    """Worker: extract metadata + embedding for one entry.

    Uses TWO separate OpenAI clients:
      - chat_client: talks to LiteLLM proxy (Claude chat models)
      - embed_client: talks to api.openai.com directly (embedding models)

    LiteLLM proxy only exposes chat models, not embedding models, so embedding
    requests MUST go to OpenAI directly.
    """
    chat_client = OpenAI(api_key=litellm_api_key, base_url=base_url)
    embed_client = OpenAI(api_key=openai_api_key)  # no base_url -> api.openai.com
    task_desc = entry.get("task_description", "") or ""
    meta = call_metadata_llm(chat_client, metadata_model, task_desc)
    emb = call_embedding(embed_client, embedding_model, task_desc)
    return idx, meta, emb


# -----------------------------
# Progressive save helpers
# -----------------------------

def _save_progress(
    enriched: List[Dict[str, Any]],
    embeddings: np.ndarray,
    output_json: str,
    output_emb: str,
) -> None:
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    # Atomic JSON write
    tmp_json = output_json + ".tmp"
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)
    os.replace(tmp_json, output_json)
    # Numpy save (atomic)
    tmp_emb = output_emb + ".tmp.npy"
    np.save(tmp_emb, embeddings)
    os.replace(tmp_emb, output_emb)


def _load_progress(
    output_json: str,
    output_emb: str,
    n_total: int,
    emb_dim: int,
) -> Tuple[List[Dict[str, Any]], np.ndarray, set]:
    """Load previously-enriched entries. Returns (enriched_list, emb_array, done_indices)."""
    enriched: List[Dict[str, Any]] = []
    emb: np.ndarray = np.zeros((n_total, emb_dim), dtype=np.float32)
    done: set = set()
    if os.path.exists(output_json) and os.path.exists(output_emb):
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                prev = json.load(f)
            prev_emb = np.load(output_emb)
            if len(prev) == n_total and prev_emb.shape == (n_total, emb_dim):
                enriched = prev
                emb = prev_emb.astype(np.float32)
                for i, e in enumerate(prev):
                    if "domain" in e and e.get("domain") is not None and np.linalg.norm(emb[i]) > 1e-6:
                        done.add(i)
                print(f"Resume: {len(done)}/{n_total} entries already enriched", flush=True)
        except Exception as e:
            print(f"Could not load prior progress ({e}); starting fresh", flush=True)
    return enriched, emb, done


# -----------------------------
# Main
# -----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-emb", required=True)
    p.add_argument("--metadata-model", default="claude-sonnet-4.6")
    p.add_argument("--embedding-model", default="text-embedding-3-small")
    p.add_argument("--emb-dim", type=int, default=1536)
    p.add_argument("--base-url", default="https://litellm.nbdevenv.xiaoaojianghu.fun",
                   help="LiteLLM proxy base URL (used ONLY for chat/metadata model).")
    p.add_argument("--api-key", default=os.environ.get("LITELLM_API_KEY", ""),
                   help="LiteLLM proxy API key (for Claude chat). Env: LITELLM_API_KEY.")
    p.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY", ""),
                   help="Direct OpenAI API key (for embedding model). Env: OPENAI_API_KEY.")
    p.add_argument("--max-workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--save-every", type=int, default=50,
                   help="Persist enriched JSON + npy every N completions")
    args = p.parse_args()

    if not args.api_key:
        print("ERROR: --api-key or $LITELLM_API_KEY required (for Claude chat via LiteLLM)")
        sys.exit(1)
    if not args.openai_api_key:
        print("ERROR: --openai-api-key or $OPENAI_API_KEY required "
              "(embeddings go DIRECTLY to api.openai.com; LiteLLM proxy does not expose embedding models)")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        source = json.load(f)
    n_total_src = len(source)
    print(f"Loaded {n_total_src} entries from {args.input}", flush=True)

    if args.limit:
        source = source[: args.limit]
    n_total = len(source)
    print(f"Processing {n_total} entries", flush=True)

    # Load prior progress if any (only when full run — limit runs always start fresh path)
    enriched, embeddings, done = _load_progress(
        args.output_json, args.output_emb, n_total, args.emb_dim
    )
    if not enriched:
        # Deep-copy-ish: shallow copy of each entry dict is fine (we add new fields)
        enriched = [dict(e) for e in source]
        embeddings = np.zeros((n_total, args.emb_dim), dtype=np.float32)
        done = set()
        # Pre-mark with default metadata so shape is stable
        for e in enriched:
            e.setdefault("domain", None)
            e.setdefault("task_type", None)
            e.setdefault("metric", None)
            e.setdefault("data_size", None)

    to_process = [i for i in range(n_total) if i not in done]
    if not to_process:
        print("All entries already enriched; nothing to do.", flush=True)
    else:
        print(f"To process: {len(to_process)} entries (skipping {len(done)} already done)", flush=True)

    t0 = time.time()
    n_done = 0
    n_err = 0
    last_save = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {
            ex.submit(
                process_one,
                i,
                source[i],
                args.metadata_model,
                args.embedding_model,
                args.base_url,
                args.api_key,
                args.openai_api_key,
            ): i
            for i in to_process
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                _, meta, emb = fut.result()
                enriched[i]["domain"] = meta["domain"]
                enriched[i]["task_type"] = meta["task_type"]
                enriched[i]["metric"] = meta["metric"]
                enriched[i]["data_size"] = meta["data_size"]
                if emb is None:
                    embeddings[i] = 0.0
                    n_err += 1
                else:
                    vec = np.asarray(emb, dtype=np.float32)
                    if vec.shape[0] != args.emb_dim:
                        print(f"  [warn] embedding dim {vec.shape[0]} != {args.emb_dim} at idx {i}", flush=True)
                        # Pad or truncate to expected dim to keep shape stable
                        if vec.shape[0] > args.emb_dim:
                            vec = vec[: args.emb_dim]
                        else:
                            padded = np.zeros(args.emb_dim, dtype=np.float32)
                            padded[: vec.shape[0]] = vec
                            vec = padded
                    embeddings[i] = vec
                done.add(i)
            except Exception as e:
                n_err += 1
                enriched[i]["domain"] = DEFAULT_METADATA["domain"]
                enriched[i]["task_type"] = DEFAULT_METADATA["task_type"]
                enriched[i]["metric"] = DEFAULT_METADATA["metric"]
                enriched[i]["data_size"] = DEFAULT_METADATA["data_size"]
                print(f"  [worker error] idx={i}: {e!s}", flush=True)
            n_done += 1
            if n_done % 10 == 0 or n_done == len(to_process):
                elapsed = time.time() - t0
                eta = elapsed / n_done * (len(to_process) - n_done) if n_done else 0
                print(
                    f"[{n_done}/{len(to_process)}] elapsed {elapsed:.0f}s eta {eta:.0f}s errors={n_err}",
                    flush=True,
                )
            if n_done - last_save >= args.save_every:
                _save_progress(enriched, embeddings, args.output_json, args.output_emb)
                last_save = n_done

    # Final save
    _save_progress(enriched, embeddings, args.output_json, args.output_emb)
    print(
        f"\nDone. Wrote {len(enriched)} enriched entries to {args.output_json}\n"
        f"Embeddings shape {embeddings.shape} -> {args.output_emb}\n"
        f"Errors: {n_err}  Total elapsed: {time.time()-t0:.1f}s",
        flush=True,
    )

    # Distribution sanity
    domain_counts = Counter(e.get("domain") for e in enriched)
    task_type_counts = Counter(e.get("task_type") for e in enriched)
    metric_counts = Counter(e.get("metric") for e in enriched)
    ds_counts = Counter(e.get("data_size") for e in enriched)
    print("\nDistribution:")
    print(f"  domain:    {dict(domain_counts)}")
    print(f"  task_type: {dict(task_type_counts)}")
    print(f"  metric:    {dict(metric_counts)}")
    print(f"  data_size: {dict(ds_counts)}")
    print(f"  embeddings dtype={embeddings.dtype} shape={embeddings.shape}")


if __name__ == "__main__":
    main()

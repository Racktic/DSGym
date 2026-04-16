"""
SmartRetriever — task-aware cross-task memory retrieval.

Consumes a pre-enriched memory file (with {domain, task_type, metric, data_size})
plus aligned float32 embeddings (.npy) produced by scripts/enrich_memory_metadata.py.

At retrieval time:
  1. Classify the current task (Claude) to get {domain, task_type, metric, data_size}
  2. Embed the current task description (text-embedding-3-small)
  3. HARD FILTER on domain + task_type match (with "other" as wildcard)
  4. Rank remaining entries by cosine similarity
  5. Optional challenge_name boost (x2) for entries sharing a long word with current challenge
  6. Per-task cap (max 2) and return top_k
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

from .memory import MemoryEntry


# -----------------------------
# Constants — mirror enrich_memory_metadata.py
# -----------------------------

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

METADATA_SYSTEM_PROMPT = (
    "You classify Kaggle-style ML task descriptions into a controlled taxonomy. "
    "Output ONLY strict JSON, no prose, no markdown."
)

METADATA_USER_TEMPLATE = """Classify this ML task into a strict JSON object with these four fields. Each field MUST use one of the listed enum values; use "other" / "unknown" when nothing fits.

{{
  "domain": one of ["tabular", "image", "text", "time_series", "graph", "audio", "multimodal", "other"],
  "task_type": one of ["binary_classification", "multiclass_classification", "multilabel_classification", "regression", "ranking", "segmentation", "detection", "generation", "other"],
  "metric": one of ["rmse", "rmsle", "mae", "auc", "log_loss", "f1", "accuracy", "ndcg", "map", "other"],
  "data_size": one of ["small", "medium", "large", "unknown"]
}}

Rules:
- Output EXACTLY one JSON object. No markdown fences, no comments.
- Use lowercase values exactly as listed above.
- If the task mixes modalities, use "multimodal".

Task description:
---
{task_description}
---

JSON:"""


def _parse_metadata_json(text: str) -> Dict[str, str]:
    if not text:
        return dict(DEFAULT_METADATA)
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
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


def _hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()


# -----------------------------
# SmartRetriever
# -----------------------------

class SmartRetriever:
    """Task-aware cross-task memory retrieval."""

    def __init__(
        self,
        enriched_json_path: str,
        embeddings_path: str,
        litellm_base_url: str,
        litellm_api_key: str,
        openai_api_key: str,
        embedding_model: str = "text-embedding-3-small",
        metadata_model: str = "claude-sonnet-4.6",
        emb_dim: int = 1536,
        retrieval_log_path: Optional[str] = None,
    ):
        if OpenAI is None:
            raise RuntimeError("openai package not installed; required for SmartRetriever")
        if not litellm_api_key:
            raise ValueError(
                "SmartRetriever: litellm_api_key missing — needed for Claude chat via LiteLLM proxy "
                "(set $LITELLM_API_KEY)"
            )
        if not openai_api_key:
            raise ValueError(
                "SmartRetriever: openai_api_key missing — embeddings go DIRECTLY to api.openai.com "
                "because the LiteLLM proxy does not expose embedding models (set $OPENAI_API_KEY)"
            )
        self.enriched_json_path = enriched_json_path
        self.embeddings_path = embeddings_path
        self.litellm_base_url = litellm_base_url
        self.litellm_api_key = litellm_api_key
        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model
        self.metadata_model = metadata_model
        self.emb_dim = emb_dim

        with open(enriched_json_path, "r", encoding="utf-8") as f:
            self.entries_raw: List[Dict[str, Any]] = json.load(f)
        self.embeddings: np.ndarray = np.load(embeddings_path).astype(np.float32)
        if self.embeddings.shape[0] != len(self.entries_raw):
            raise ValueError(
                f"Embedding count {self.embeddings.shape[0]} != entry count {len(self.entries_raw)}"
            )
        # L2-normalize for cosine via dot product
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        self.embeddings_normed: np.ndarray = self.embeddings / norms

        # Two separate clients: LiteLLM proxy for Claude chat, direct OpenAI for embeddings.
        self._chat_client = OpenAI(api_key=litellm_api_key, base_url=litellm_base_url)
        self._embed_client = OpenAI(api_key=openai_api_key)  # no base_url -> api.openai.com

        # Optional debug log: one JSONL line per (task, action) retrieval with top-K cosines.
        # Used offline to pick a similarity threshold.
        self.retrieval_log_path: Optional[str] = retrieval_log_path
        if retrieval_log_path:
            os.makedirs(os.path.dirname(os.path.abspath(retrieval_log_path)) or ".", exist_ok=True)

        # Caches
        self._meta_cache: Dict[str, Dict[str, str]] = {}
        self._emb_cache: Dict[str, np.ndarray] = {}
        # (task_desc_hash, action) -> ordered list of entry indices (pool for random sampling)
        self._pool_cache: Dict[tuple, List[int]] = {}
        self._cache_lock = threading.Lock()
        # Independent RNG so sampling is reproducible and not coupled to caller seed
        import random as _random
        self._rng = _random.Random(0xD55)

    # ---------------- classification + embedding ----------------

    def _classify_task(self, task_description: str) -> Dict[str, str]:
        key = _hash_text(task_description)
        with self._cache_lock:
            cached = self._meta_cache.get(key)
        if cached is not None:
            return cached

        # Full description — useful fields (data/metric/submission) are mid/late.
        prompt = METADATA_USER_TEMPLATE.format(task_description=task_description or "")
        meta = dict(DEFAULT_METADATA)
        for attempt in range(3):
            try:
                resp = self._chat_client.chat.completions.create(
                    model=self.metadata_model,
                    messages=[
                        {"role": "system", "content": METADATA_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=200,
                )
                meta = _parse_metadata_json(resp.choices[0].message.content)
                break
            except Exception as e:
                time.sleep(2 * (attempt + 1))
                if attempt == 2:
                    print(f"[SmartRetriever] classify failed, using defaults: {e!s}")
        with self._cache_lock:
            self._meta_cache[key] = meta
        return meta

    def _embed_task(self, task_description: str) -> np.ndarray:
        key = _hash_text(task_description)
        with self._cache_lock:
            cached = self._emb_cache.get(key)
        if cached is not None:
            return cached
        t = task_description or ""
        # Token-aware truncation to 8000 tokens (text-embedding-3-small hard limit is 8191)
        if t:
            try:
                import tiktoken
                _enc = tiktoken.get_encoding("cl100k_base")
                _tok = _enc.encode(t)
                if len(_tok) > 8000:
                    t = _enc.decode(_tok[:8000])
            except Exception:
                if len(t) > 25000:
                    t = t[:25000]
        vec = np.zeros(self.emb_dim, dtype=np.float32)
        if t:
            for attempt in range(3):
                try:
                    resp = self._embed_client.embeddings.create(model=self.embedding_model, input=t)
                    v = np.asarray(resp.data[0].embedding, dtype=np.float32)
                    if v.shape[0] != self.emb_dim:
                        if v.shape[0] > self.emb_dim:
                            v = v[: self.emb_dim]
                        else:
                            padded = np.zeros(self.emb_dim, dtype=np.float32)
                            padded[: v.shape[0]] = v
                            v = padded
                    vec = v
                    break
                except Exception as e:
                    time.sleep(2 * (attempt + 1))
                    if attempt == 2:
                        print(f"[SmartRetriever] embed failed, using zeros: {e!s}")
        n = np.linalg.norm(vec)
        if n > 1e-8:
            vec = vec / n
        with self._cache_lock:
            self._emb_cache[key] = vec
        return vec

    # ---------------- retrieval ----------------

    @staticmethod
    def _entry_type_weight(entry_type: str, current_action: str) -> float:
        """DEPRECATED — not currently wired into ranking.

        Intent: soft-boost entries whose `entry_type` matches the current agent
        action (e.g., during `improve`, prefer `improvement` entries over others).

        Why unused: `_build_pool` now applies a HARD filter via `required_type`
        (action→entry_type: draft→draft_success, improve→improvement,
        debug→debug_fix). That filter reduces the pool to exactly one entry_type,
        which makes this soft weight redundant. To re-enable soft ranking,
        first remove the `required_type` filter in `_build_pool`.

        Kept here as a placeholder for a future soft-ranking variant.
        """
        if current_action in ("debug",):
            table = {"debug_fix": 3.0, "improvement": 1.5, "task_summary": 1.0, "draft_success": 1.0}
        elif current_action in ("improve", "exploit"):
            table = {"improvement": 3.0, "task_summary": 1.5, "draft_success": 1.5, "debug_fix": 1.0}
        else:  # draft / explore / unknown
            table = {"task_summary": 3.0, "improvement": 1.5, "draft_success": 2.0, "debug_fix": 1.0}
        return table.get(entry_type, 0.5)

    @staticmethod
    def _compatible(cur: str, stored: str) -> bool:
        """Hard-filter compatibility: match OR either side is 'other'/'unknown'."""
        if cur == stored:
            return True
        if cur in ("other", "unknown") or stored in ("other", "unknown"):
            return True
        return False

    @staticmethod
    def _share_long_word(a: str, b: str, min_len: int = 4) -> bool:
        if not a or not b:
            return False
        import re
        tok = lambda s: set(w for w in re.split(r"[^a-zA-Z0-9]+", s.lower()) if len(w) > min_len - 1)
        return bool(tok(a) & tok(b))

    def _build_pool(
        self,
        current_task_description: str,
        current_action: str,
        challenge_name: str,
        pool_size: int,
        candidate_pool: int,
        max_per_task: int,
        challenge_name_boost: bool,
    ) -> List[int]:
        """Build a ranked pool of up to `pool_size` entry indices for this (task, action).
        Cached per (task_desc_hash, action) so we pay the API cost only once per task/phase."""
        task_hash = _hash_text(current_task_description)
        cache_key = (task_hash, current_action)
        with self._cache_lock:
            cached = self._pool_cache.get(cache_key)
        if cached is not None:
            return cached

        cur_meta = self._classify_task(current_task_description)
        q_vec = self._embed_task(current_task_description)
        n = len(self.entries_raw)

        action_to_type = {
            "draft": "draft_success",
            "improve": "improvement",
            "debug": "debug_fix",
        }
        required_type = action_to_type.get(current_action)
        keep_mask = np.zeros(n, dtype=bool)
        for i, e in enumerate(self.entries_raw):
            if e.get("challenge_name", "") == challenge_name and challenge_name:
                continue  # exclude current task
            if required_type is not None and e.get("entry_type") != required_type:
                continue
            d = (e.get("domain") or "other").lower()
            tt = (e.get("task_type") or "other").lower()
            if not self._compatible(cur_meta["domain"], d):
                continue
            if not self._compatible(cur_meta["task_type"], tt):
                continue
            keep_mask[i] = True
        kept_idx = np.nonzero(keep_mask)[0]
        if kept_idx.size == 0:
            with self._cache_lock:
                self._pool_cache[cache_key] = []
            return []

        sub_emb = self.embeddings_normed[kept_idx]
        cosines = sub_emb @ q_vec

        pool_n = min(candidate_pool, cosines.shape[0])
        top_local = np.argpartition(-cosines, pool_n - 1)[:pool_n]
        top_local = top_local[np.argsort(-cosines[top_local])]
        cand_idx = kept_idx[top_local]
        cand_cos = cosines[top_local]

        rerank = []
        for pos, i in enumerate(cand_idx):
            e = self.entries_raw[i]
            s = float(cand_cos[pos])
            if challenge_name_boost and challenge_name:
                if self._share_long_word(challenge_name, e.get("challenge_name", ""), min_len=4):
                    s *= 2.0
            rerank.append((s, i))

        rerank.sort(key=lambda x: x[0], reverse=True)

        # Debug dump: one JSONL line per (task, action) retrieval.
        if self.retrieval_log_path:
            top_k_log = 10
            log_entries = []
            for pos, i in enumerate(cand_idx[:top_k_log]):
                e = self.entries_raw[i]
                log_entries.append({
                    "challenge_name": e.get("challenge_name", ""),
                    "entry_type": e.get("entry_type", ""),
                    "cosine": float(cand_cos[pos]),
                    "domain": e.get("domain", ""),
                    "task_type": e.get("task_type", ""),
                })
            record = {
                "task_desc_hash": task_hash,
                "challenge_name": challenge_name,
                "current_action": current_action,
                "classified_meta": cur_meta,
                "kept_pool_size": int(kept_idx.size),
                "top_k": log_entries,
            }
            try:
                with self._cache_lock:  # serialize appends across worker threads
                    with open(self.retrieval_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[SmartRetriever] retrieval_log write failed: {e!s}")

        # Per-task cap, then take top `pool_size`
        pool: List[int] = []
        per_task: Dict[str, int] = {}
        for _, i in rerank:
            e = self.entries_raw[i]
            cn = e.get("challenge_name", "")
            if per_task.get(cn, 0) >= max_per_task:
                continue
            pool.append(int(i))
            per_task[cn] = per_task.get(cn, 0) + 1
            if len(pool) >= pool_size:
                break

        with self._cache_lock:
            self._pool_cache[cache_key] = pool
        return pool

    def retrieve(
        self,
        current_task_description: str,
        current_action: str = "",
        challenge_name: str = "",
        top_k: int = 3,
        pool_size: int = 15,
        challenge_name_boost: bool = False,
        candidate_pool: int = 60,
        max_per_task: int = 2,
        random_sample: bool = True,
    ) -> List[MemoryEntry]:
        """Retrieve up to `top_k` memory entries for this turn.

        Strategy (simplest baseline):
          1. Per (task, action) build + cache a ranked pool of `pool_size` entries
          2. Each call randomly samples `top_k` from that pool

        This keeps the pool stable (same 15 per task/action across turns) while
        varying which 3 the agent sees each turn, at zero extra retrieval cost.
        """
        pool = self._build_pool(
            current_task_description=current_task_description,
            current_action=current_action,
            challenge_name=challenge_name,
            pool_size=pool_size,
            candidate_pool=candidate_pool,
            max_per_task=max_per_task,
            challenge_name_boost=challenge_name_boost,
        )
        if not pool:
            return []

        if random_sample and len(pool) > top_k:
            selected = self._rng.sample(pool, top_k)
        else:
            selected = pool[:top_k]
        return [MemoryEntry.from_dict(self.entries_raw[i]) for i in selected]

    # ---------------- formatting ----------------

    @staticmethod
    def format_for_prompt(entries: List[MemoryEntry]) -> str:
        """Prompt-ready block, mirrors CrossTaskMemory.format_for_prompt shape."""
        if not entries:
            return ""
        lines = [
            "=== CROSS-TASK EXPERIENCE MEMORY ===",
            "Below are key experiences from solving OTHER similar tasks, ranked by task "
            "similarity and relevance. Use these to inform your approach.",
            "",
        ]
        by_challenge: Dict[str, List[MemoryEntry]] = {}
        for entry in entries:
            by_challenge.setdefault(entry.challenge_name, []).append(entry)
        for cname, c_entries in by_challenge.items():
            lines.append(f"--- Task: {cname} ---")
            for entry in c_entries:
                tag = {
                    "task_summary": "SUMMARY",
                    "improvement": "IMPROVED",
                    "debug_fix": "DEBUG_FIX",
                    "draft_success": "DRAFT",
                }.get(entry.entry_type, "INFO")
                score_str = f"{entry.score:.6f}" if entry.score is not None else "N/A"
                lines.append(
                    f"  [{tag}] Model: {entry.model_type} | "
                    f"Score: {score_str} | {entry.insight}"
                )
            lines.append("")
        lines.append("=== END CROSS-TASK MEMORY ===")
        return "\n".join(lines)

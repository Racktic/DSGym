"""
Local evaluation metric for DSPredict target-swap tasks.

Evaluates submission.csv against ground_truth.csv using the metric
specified in each task's metadata (RMSE, RMSLE, AUC, LogLoss, Accuracy).
"""

from __future__ import annotations

import os
import time
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from ..base import BaseMetric, MetricResult
from dsgym.datasets.config import REPO_ROOT


SWAP_GT_DIR = REPO_ROOT / "data" / "data" / "dspredict-swap-ground-truth"


def _compute_metric(y_true: np.ndarray, y_pred: np.ndarray, metric_name: str) -> float:
    """Compute a specific metric between true and predicted values."""
    if metric_name == "rmse":
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    elif metric_name == "rmsle":
        # Clip to avoid log of negative
        y_true_c = np.clip(y_true, 0, None)
        y_pred_c = np.clip(y_pred, 0, None)
        return float(np.sqrt(np.mean((np.log1p(y_true_c) - np.log1p(y_pred_c)) ** 2)))

    elif metric_name == "auc":
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_pred))

    elif metric_name == "log_loss":
        from sklearn.metrics import log_loss
        return float(log_loss(y_true, y_pred))

    elif metric_name == "accuracy":
        from sklearn.metrics import accuracy_score
        return float(accuracy_score(y_true, y_pred))

    else:
        # Default to RMSE
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _is_lower_better(metric_name: str) -> bool:
    """Return True if lower scores are better."""
    return metric_name in ("rmse", "rmsle", "log_loss")


class SwapSubmissionMetric(BaseMetric):
    """Metric for DSPredict swap tasks — local evaluation using ground_truth.csv."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._baseline_scores = None  # Lazy-loaded

    @property
    def name(self) -> str:
        return "swap_submission"

    @property
    def requires_ground_truth(self) -> bool:
        return False  # Ground truth is loaded from file, not passed in

    def evaluate(
        self,
        prediction: str,
        ground_truth: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs,
    ) -> MetricResult:
        start = time.time()
        extra_info = kwargs.get("extra_info", {})
        challenge_name = extra_info.get("challenge_name", "")

        # Get metric type from metadata
        metadata = extra_info.get("metadata", {})
        metric_name = metadata.get("metric", "rmse")
        task_type = metadata.get("task_type", "regression")
        swap_target = metadata.get("swap_target", "")

        # Find submission file
        submission_path = (prediction or "").strip()
        if not submission_path or not os.path.exists(submission_path):
            return MetricResult(
                metric_name=self.name, score=None,
                details={"reason": "submission file not found", "prediction": prediction},
                evaluation_time=time.time() - start,
            )

        # Find ground truth file
        gt_path = SWAP_GT_DIR / challenge_name / "ground_truth.csv"
        if not gt_path.exists():
            return MetricResult(
                metric_name=self.name, score=None,
                details={"reason": f"ground truth not found at {gt_path}"},
                evaluation_time=time.time() - start,
            )

        try:
            sub_df = pd.read_csv(submission_path)
            gt_df = pd.read_csv(gt_path)

            # Align by ID column (first column)
            id_col = gt_df.columns[0]
            target_col = gt_df.columns[1]

            # Merge on ID
            merged = gt_df.merge(sub_df, on=id_col, how="inner", suffixes=("_true", "_pred"))

            if len(merged) == 0:
                return MetricResult(
                    metric_name=self.name, score=None,
                    details={"reason": "no matching IDs between submission and ground truth"},
                    evaluation_time=time.time() - start,
                )

            # Find the predicted column
            if f"{target_col}_pred" in merged.columns:
                pred_col = f"{target_col}_pred"
                true_col = f"{target_col}_true"
            elif target_col in sub_df.columns:
                # If sub_df had same column name, merge added suffix to gt
                pred_col = target_col
                true_col = f"{target_col}_true" if f"{target_col}_true" in merged.columns else target_col
            else:
                # Try second column of submission
                pred_col = sub_df.columns[1]
                true_col = f"{target_col}_true" if f"{target_col}_true" in merged.columns else target_col

            y_true = merged[true_col].values.astype(float)
            y_pred = merged[pred_col].values.astype(float)

            score = _compute_metric(y_true, y_pred, metric_name)
            lower_better = _is_lower_better(metric_name)

            details = {
                "status": "COMPLETE",
                "challenge_name": challenge_name,
                "metric_type": metric_name,
                "task_type": task_type,
                "swap_target": swap_target,
                "score": score,
                "lower_is_better": lower_better,
                "n_samples_evaluated": len(merged),
                "n_samples_ground_truth": len(gt_df),
                "local_submission_file": submission_path,
            }

            return MetricResult(
                metric_name=self.name,
                score=score,
                details=details,
                evaluation_time=time.time() - start,
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return MetricResult(
                metric_name=self.name, score=None,
                error=str(e),
                details={"challenge_name": challenge_name},
                evaluation_time=time.time() - start,
            )

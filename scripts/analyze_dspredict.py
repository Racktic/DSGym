#!/usr/bin/env python3
"""
DSPredict 评测结果分析脚本

计算论文中的三个核心指标:
  - Valid: 成功提交并拿到Kaggle分数的比例
  - Percentile: 有效提交在Kaggle排行榜上的平均百分位排名
  - Above Median: 有效提交中超过排行榜中位数的比例

用法:
    python scripts/analyze_dspredict.py <results_dir>
    python scripts/analyze_dspredict.py results/dspredict_easy_full_v3
"""

import json
import sys
import glob
import os


def analyze(results_dir: str):
    # 找到 *_results.json 文件
    result_files = glob.glob(os.path.join(results_dir, "*_results.json"))
    if not result_files:
        print(f"错误: 在 {results_dir} 下没有找到 *_results.json 文件")
        sys.exit(1)

    results_file = result_files[0]
    print(f"分析文件: {results_file}\n")

    with open(results_file) as f:
        data = json.load(f)

    results = data if isinstance(data, list) else data.get("results", [])
    total = len(results)

    # 收集每个sample的指标
    samples = []
    for r in results:
        sample_id = r.get("sample_id", "")
        # 取最后一段作为短名
        short_name = sample_id.split("_")[-1] if "_" in sample_id else sample_id

        metrics = r.get("metrics", {})
        for k, v in metrics.items():
            details = v.get("details", {})
            samples.append({
                "name": short_name,
                "status": details.get("status"),
                "score": v.get("score"),
                "public_score": details.get("public_score"),
                "private_score": details.get("private_score"),
                "public_percentile": details.get("public_percentile"),
                "private_percentile": details.get("private_percentile"),
                "public_above_median": details.get("public_above_median"),
                "private_above_median": details.get("private_above_median"),
                "public_rank": details.get("public_rank"),
                "public_medal": details.get("public_medal"),
            })

    # 分类统计
    complete = [s for s in samples if s["status"] == "SUBMISSIONSTATUS.COMPLETE"]
    error = [s for s in samples if s["status"] == "SUBMISSIONSTATUS.ERROR"]
    no_submission = [s for s in samples if s["status"] is None]

    valid_rate = len(complete) / total if total > 0 else 0

    public_percentiles = [s["public_percentile"] for s in complete if s["public_percentile"] is not None]
    private_percentiles = [s["private_percentile"] for s in complete if s["private_percentile"] is not None]

    public_above = [s for s in complete if s["public_above_median"] is True]
    private_above = [s for s in complete if s["private_above_median"] is True]

    avg_public_percentile = sum(public_percentiles) / len(public_percentiles) if public_percentiles else 0
    avg_private_percentile = sum(private_percentiles) / len(private_percentiles) if private_percentiles else 0

    above_median_rate = len(public_above) / total if total > 0 else 0
    private_above_median_rate = len(private_above) / total if total > 0 else 0

    # ============ 输出汇总 ============
    print("=" * 60)
    print("DSPredict 评测结果汇总")
    print("=" * 60)

    print(f"\n总样本数: {total}")
    print(f"  成功提交 (COMPLETE): {len(complete)}")
    print(f"  提交失败 (ERROR):    {len(error)}")
    print(f"  未生成提交:          {len(no_submission)}")

    print(f"\n{'指标':<25} {'Public':<15} {'Private':<15}")
    print("-" * 55)
    print(f"{'Valid (成功提交率)':<25} {valid_rate:<15.1%}")
    print(f"{'Avg Percentile (平均百分位)':<25} {avg_public_percentile:<15.2f} {avg_private_percentile:<15.2f}")
    print(f"{'Above Median (超中位数率)':<25} {above_median_rate:<15.1%} {private_above_median_rate:<15.1%}")

    # ============ 每个sample的详情 ============
    print(f"\n{'Sample':<45} {'Status':<10} {'Public Score':<15} {'Percentile':<12} {'> Median':<10}")
    print("-" * 92)
    for s in samples:
        status_short = {
            "SUBMISSIONSTATUS.COMPLETE": "OK",
            "SUBMISSIONSTATUS.ERROR": "ERROR",
            None: "N/A",
        }.get(s["status"], s["status"])

        score_str = f"{s['public_score']}" if s["public_score"] else "-"
        pct_str = f"{s['public_percentile']:.1f}%" if s["public_percentile"] is not None else "-"
        median_str = "Yes" if s["public_above_median"] is True else ("No" if s["public_above_median"] is False else "-")

        print(f"{s['name']:<45} {status_short:<10} {score_str:<15} {pct_str:<12} {median_str:<10}")

    # ============ 输出论文格式的结果 ============
    print(f"\n论文指标 (Public Leaderboard):")
    print(f"  Valid:      {valid_rate:.1%} ({len(complete)}/{total})")
    print(f"  Percentile: {avg_public_percentile:.2f}")
    print(f"  Median:     {above_median_rate:.1%} ({len(public_above)}/{total})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    analyze(sys.argv[1])

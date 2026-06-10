"""
Week 3: 逐缺陷类型分析 (从机械视角解读)。
用 PaDiM (CPU 快) 对 metal_nut 测试集逐张打异常分数, 按缺陷类型聚合:
样本数、平均异常分、检测率 (用 good 样本分数的统计量定阈值)。

输出: results/defect_analysis.md —— 回答"模型对哪类缺陷更难/更易, 为什么"。
运行: set PYTHONUTF8=1 && python scripts/run_defect_analysis.py
"""
from collections import defaultdict
from pathlib import Path

import torch
from anomalib.data import MVTecAD
from anomalib.models import Padim
from anomalib.engine import Engine

ROOT = Path(__file__).resolve().parent.parent
CATEGORY = "metal_nut"


def defect_type_from_path(p: str) -> str:
    """test/<defect_type>/000.png -> <defect_type>"""
    return Path(p).parent.name


if __name__ == "__main__":
    torch.manual_seed(0)
    dm = MVTecAD(
        root=str(ROOT / "datasets" / "MVTecAD"), category=CATEGORY,
        train_batch_size=8, eval_batch_size=8, num_workers=0,
    )
    model = Padim()
    engine = Engine(
        default_root_dir=str(ROOT / "results"),
        accelerator="cpu", devices=1, enable_progress_bar=False,
    )
    engine.fit(datamodule=dm, model=model)

    # 逐张预测, 收集 (异常分数, 缺陷类型)
    preds = engine.predict(datamodule=dm, model=model)
    scores_by_type = defaultdict(list)
    for batch in preds:
        pred_scores = batch.pred_score
        paths = batch.image_path
        for score, path in zip(pred_scores, paths):
            t = defect_type_from_path(path)
            scores_by_type[t].append(float(score))

    # 用 good 的分数分布定阈值: mean + 2*std (经验上界, 偏向高召回可下调)
    import statistics as st
    good = scores_by_type.get("good", [])
    if good:
        thr = st.mean(good) + 2 * (st.pstdev(good) or 1e-6)
    else:
        thr = 0.5
    print(f"good 样本: n={len(good)}, mean={st.mean(good):.3f}, 阈值(mean+2std)={thr:.3f}")

    # 汇总每类
    rows = []
    for t in sorted(scores_by_type):
        s = scores_by_type[t]
        if t == "good":
            # good 类: 检测率指"误报率"(被判为异常的比例)
            detected = sum(1 for x in s if x > thr) / len(s)
            rows.append((t, len(s), st.mean(s), min(s), max(s), f"{detected:.0%} (误报)"))
        else:
            detected = sum(1 for x in s if x > thr) / len(s)
            rows.append((t, len(s), st.mean(s), min(s), max(s), f"{detected:.0%}"))

    # 写 markdown
    lines = [
        "# 逐缺陷类型分析 (metal_nut, PaDiM)",
        "",
        f"图像级异常分阈值 = good 样本 mean + 2·std = **{thr:.3f}** (偏向高召回可下调)。",
        "",
        "| 缺陷类型 | 样本数 | 平均异常分 | 最小 | 最大 | 检出率 |",
        "|---|---|---|---|---|---|",
    ]
    for t, n, mean, mn, mx, det in rows:
        lines.append(f"| {t} | {n} | {mean:.3f} | {mn:.3f} | {mx:.3f} | {det} |")
    out = "\n".join(lines) + "\n"
    (ROOT / "results" / "defect_analysis.md").write_text(out, encoding="utf-8")

    print("\n" + out)
    print("已写入 results/defect_analysis.md")

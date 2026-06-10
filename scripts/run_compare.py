"""
Week 2: 多模型对比 (PaDiM vs PatchCore), 全程 CPU。
对每个模型测: Image/Pixel AUROC + F1、拟合耗时、单张推理延迟、checkpoint 体积。
结果写到 results/comparison.json 和 results/comparison.md。

选型说明: PaDiM / PatchCore 都是"特征型"模型, 无需反向传播训练, 适合 CPU。
EfficientAD 需要真正训练 (CPU 太慢), 留到有 GPU 再补。

依赖坑见 run_padim.py 注释: pandas<3 + 运行时设 PYTHONUTF8=1。
"""
import json
import time
from pathlib import Path

import torch
from anomalib.data import MVTecAD
from anomalib.models import Padim, Patchcore
from anomalib.engine import Engine

ROOT = Path(__file__).resolve().parent.parent
CATEGORY = "metal_nut"
DATA_ROOT = str(ROOT / "datasets" / "MVTecAD")

MODELS = {"PaDiM": Padim, "PatchCore": Patchcore}


def latest_ckpt_size_mb(model_name: str) -> float:
    """找该模型最近一次 run 的 .ckpt 体积 (MB)。"""
    base = ROOT / "results" / model_name / "MVTecAD" / CATEGORY
    ckpts = list(base.glob("**/*.ckpt"))
    if not ckpts:
        return float("nan")
    newest = max(ckpts, key=lambda p: p.stat().st_mtime)
    return newest.stat().st_size / 1024 / 1024


def measure_latency_ms(engine: Engine, model, datamodule) -> float:
    """用 engine.predict 跑完整测试集, 算每张平均端到端延迟 (ms)。"""
    n = len(datamodule.test_data)
    t0 = time.perf_counter()
    engine.predict(datamodule=datamodule, model=model)
    dt = time.perf_counter() - t0
    return dt / max(n, 1) * 1000.0


def run_one(name: str, ModelCls) -> dict:
    print(f"\n===== {name} =====")
    dm = MVTecAD(
        root=DATA_ROOT, category=CATEGORY,
        train_batch_size=8, eval_batch_size=8, num_workers=0,
    )
    model = ModelCls()
    engine = Engine(
        default_root_dir=str(ROOT / "results"),
        accelerator="cpu", devices=1, enable_progress_bar=False,
    )

    print(f"[{name}] 拟合...")
    t0 = time.perf_counter()
    engine.fit(datamodule=dm, model=model)
    fit_s = time.perf_counter() - t0

    print(f"[{name}] 评测...")
    test_out = engine.test(datamodule=dm, model=model)
    metrics = dict(test_out[0]) if test_out else {}

    print(f"[{name}] 测推理延迟...")
    lat_ms = measure_latency_ms(engine, model, dm)

    return {
        "model": name,
        "image_AUROC": round(float(metrics.get("image_AUROC", float("nan"))), 4),
        "pixel_AUROC": round(float(metrics.get("pixel_AUROC", float("nan"))), 4),
        "image_F1": round(float(metrics.get("image_F1Score", float("nan"))), 4),
        "pixel_F1": round(float(metrics.get("pixel_F1Score", float("nan"))), 4),
        "fit_seconds": round(fit_s, 1),
        "latency_ms_per_image": round(lat_ms, 1),
        "ckpt_size_MB": round(latest_ckpt_size_mb(name), 2),
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    rows = [run_one(name, cls) for name, cls in MODELS.items()]

    # 写 JSON
    (ROOT / "results" / "comparison.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 写 Markdown 表
    headers = ["model", "image_AUROC", "pixel_AUROC", "image_F1",
               "pixel_F1", "fit_seconds", "latency_ms_per_image", "ckpt_size_MB"]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
    table = "\n".join(lines)
    (ROOT / "results" / "comparison.md").write_text(table + "\n", encoding="utf-8")

    print("\n========== 对比结果 ==========")
    print(table)
    print("\n完成! 见 results/comparison.json 和 results/comparison.md")

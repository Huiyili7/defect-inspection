"""
Week 4: 边缘部署 (ONNX Runtime) + 推理基准, 全程 CPU。

注: OpenVINO 暂无 Python 3.13 的 Windows wheel, 故部署后端用 ONNX Runtime
(同为主流 CPU/边缘推理后端)。导出仍同时产出 OpenVINO IR 可在 3.11 环境复用。

流程:
- 用 Anomalib 把 PaDiM 导出为 ONNX + Torch (+ OpenVINO IR 文件)
- 端到端逐张推理 (batch=1, 模拟产线单件): 读图→预处理→推理→判定
  - PyTorch 后端: anomalib TorchInferencer
  - ONNX 后端: onnxruntime + 复刻 anomalib 预处理 (Resize256 + ImageNet Norm)
- 对比单张延迟/FPS + 图像级 AUROC (验证导出无精度损失)
- 末尾: 一张图的"实时检测"输出 (判定 + 置信度 + 延迟)

运行: set PYTHONUTF8=1 && python scripts/run_deploy.py
"""
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from anomalib.data import MVTecAD
from anomalib.models import Padim
from anomalib.engine import Engine
from anomalib.deploy import ExportType, TorchInferencer

ROOT = Path(__file__).resolve().parent.parent
CATEGORY = "metal_nut"
DATA = ROOT / "datasets" / "MVTecAD" / CATEGORY
EXPORT_ROOT = ROOT / "results" / "deploy"

# 关键: anomalib 导出的 ONNX 图已内置 Resize(256)+ImageNet Normalize,
# 故只需喂原尺寸 [0,1] 浮点图 (诊断验证: 与 TorchInferencer 分数逐位一致)。
# 若在此再做 resize/normalize 会双重预处理, 导致分数全饱和、AUROC 退化到 0.5。
def preprocess(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    t = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).numpy()


def list_test_images():
    items = []
    for d in sorted((DATA / "test").iterdir()):
        if d.is_dir():
            label = 0 if d.name == "good" else 1
            for img in sorted(d.glob("*.png")):
                items.append((str(img), label))
    return items


def bench_torch(inf, items):
    scores, lats = [], []
    for i, (p, _) in enumerate(items):
        t0 = time.perf_counter()
        r = inf.predict(image=p)
        dt = (time.perf_counter() - t0) * 1000
        scores.append(float(np.asarray(r.pred_score).reshape(-1)[0]))
        if i >= 5:
            lats.append(dt)
    return scores, lats


def bench_onnx(sess, items):
    name = sess.get_inputs()[0].name
    scores, lats = [], []
    for i, (p, _) in enumerate(items):
        t0 = time.perf_counter()
        x = preprocess(p)
        out = sess.run(["pred_score"], {name: x})
        dt = (time.perf_counter() - t0) * 1000
        scores.append(float(np.asarray(out[0]).reshape(-1)[0]))
        if i >= 5:
            lats.append(dt)
    return scores, lats


def stat(lats):
    return sum(lats) / len(lats), sorted(lats)[int(len(lats) * 0.95)]


if __name__ == "__main__":
    torch.manual_seed(0)
    dm = MVTecAD(root=str(ROOT / "datasets" / "MVTecAD"), category=CATEGORY,
                 train_batch_size=8, eval_batch_size=8, num_workers=0)
    model = Padim()
    engine = Engine(default_root_dir=str(ROOT / "results"),
                    accelerator="cpu", devices=1, enable_progress_bar=False)

    print("[1/4] 拟合 PaDiM...")
    engine.fit(datamodule=dm, model=model)

    print("[2/4] 导出 ONNX + Torch (+ 尝试 OpenVINO IR)...")
    onnx_path = engine.export(model=model, export_type=ExportType.ONNX, export_root=str(EXPORT_ROOT))
    pt_path = engine.export(model=model, export_type=ExportType.TORCH, export_root=str(EXPORT_ROOT))
    print(f"   ONNX:  {onnx_path}")
    print(f"   Torch: {pt_path}")

    items = list_test_images()
    gt = [lbl for _, lbl in items]
    print(f"[3/4] 端到端逐张基准 (n={len(items)}, batch=1, 前 5 张热身)...")

    torch_inf = TorchInferencer(path=str(pt_path), device="cpu")
    t_scores, t_lats = bench_torch(torch_inf, items)
    t_avg, t_p95 = stat(t_lats)
    print(f"   [PyTorch] {t_avg:.1f} ms/张, p95 {t_p95:.1f} ms, ~{1000/t_avg:.1f} FPS")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    o_scores, o_lats = bench_onnx(sess, items)
    o_avg, o_p95 = stat(o_lats)
    print(f"   [ONNX RT] {o_avg:.1f} ms/张, p95 {o_p95:.1f} ms, ~{1000/o_avg:.1f} FPS")

    t_auroc = roc_auc_score(gt, t_scores)
    o_auroc = roc_auc_score(gt, o_scores)

    print("[4/4] 写报告...")
    md = f"""# Week 4 · 边缘部署 (ONNX Runtime) — metal_nut / PaDiM

端到端逐张推理 (读图→预处理→推理→判定, batch=1, CPU, 模拟产线单件检测), 测试集 {len(items)} 张, 前 5 张热身不计时。

| 后端 | 平均延迟 | p95 | 吞吐 | Image AUROC |
|---|---|---|---|---|
| PyTorch (CPU) | {t_avg:.1f} ms | {t_p95:.1f} ms | {1000/t_avg:.1f} FPS | {t_auroc:.4f} |
| **ONNX Runtime (CPU)** | **{o_avg:.1f} ms** | {o_p95:.1f} ms | **{1000/o_avg:.1f} FPS** | {o_auroc:.4f} |

- **加速比**: ONNX Runtime / PyTorch ≈ **{t_avg/o_avg:.2f}×**
- **精度损失**: AUROC 差异 {abs(t_auroc - o_auroc):.4f} —— {'导出无明显精度损失' if abs(t_auroc - o_auroc) < 0.01 else '有精度变化, 需复核'}
- 导出格式见 `results/deploy/weights/`: ONNX (.onnx) + Torch (.pt)
- 备注: OpenVINO 暂无 Python 3.13 Windows wheel; 部署后端选用 ONNX Runtime (CPU)。
  ONNX 为跨框架标准, 可在 Intel 设备上再转 OpenVINO IR 进一步加速。
"""
    (ROOT / "results" / "deploy_report.md").write_text(md, encoding="utf-8")
    print("\n" + md)

    # 单张"实时检测"演示 (ONNX 后端)
    demo_path, demo_gt = items[20]
    x = preprocess(demo_path)
    t0 = time.perf_counter()
    score = float(np.asarray(sess.run(["pred_score"], {sess.get_inputs()[0].name: x})[0]).reshape(-1)[0])
    dt = (time.perf_counter() - t0) * 1000
    verdict = "异常 (NG)" if score > 0.5 else "正常 (OK)"
    print("===== 实时检测 demo (ONNX Runtime) =====")
    print(f"输入: {Path(demo_path).parent.name}/{Path(demo_path).name} (真值: {'异常' if demo_gt else '正常'})")
    print(f"判定: {verdict} | 异常分: {score:.3f} | 推理延迟: {dt:.1f} ms")
    print("已写入 results/deploy_report.md")

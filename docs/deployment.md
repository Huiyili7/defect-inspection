# Week 4 · 边缘部署(ONNX Runtime)

把训练好的 PaDiM 导出为部署格式,在 CPU 上做"产线单件检测"式的端到端推理,并验证导出未损失精度。

## 后端选型说明

原计划用 **OpenVINO**(Intel 的 CPU 推理工具,最贴合边缘/嵌入式)。但 OpenVINO 当前**没有 Python 3.13 的 Windows wheel**,装不上。故部署后端改用 **ONNX Runtime**——同为主流 CPU 推理后端,ONNX 又是跨框架标准格式,后续可在 Intel 设备上再转 OpenVINO IR 进一步加速。导出脚本同时产出 ONNX 与 Torch 两种格式。

## 推理基准(端到端:读图 → 预处理 → 推理 → 判定)

`batch=1`(模拟产线逐件检测),CPU,测试集 115 张,前 5 张热身不计时:

| 后端 | 平均延迟 | p95 | 吞吐 | Image AUROC |
|---|---|---|---|---|
| PyTorch (CPU) | 189.4 ms | 215.1 ms | 5.3 FPS | 0.9482 |
| **ONNX Runtime (CPU)** | **132.9 ms** | 189.5 ms | **7.5 FPS** | **0.9482** |

- **加速比**:ONNX Runtime / PyTorch ≈ **1.43×**(同一 CPU、同一精度下)
- **精度对齐**:两者 Image AUROC 完全相同(差异 0.0000)→ **导出无精度损失**
- **实时 demo**:单张 bent 缺陷图 → 判定「异常(NG)」,异常分 0.702,推理 ~72 ms

## 一个关键的工程坑:不要双重预处理

Anomalib 导出的 ONNX 图**已把预处理(Resize 256 + ImageNet Normalize)打包进图内**,输入是原尺寸 `[0,1]` 浮点图。我最初在喂图前又手动做了一遍 Resize+Normalize,导致**双重预处理**——所有图(含正常件)都被推到"分布外",异常分全部饱和到 1.0,**ONNX 的 AUROC 退化到 0.5(随机)**。

定位方法:同一张图分别用「原图[0,1]」「原图[0,255]」「resize+norm」喂 ONNX,与 `TorchInferencer` 的分数逐一对照——只有「原图[0,1]」能**逐位复现** TorchInferencer 的分数(good 0.5131、bent 0.6994)。修正后 ONNX 与 PyTorch 精度完全对齐。

> 教训:验证模型导出正确性时,不能只看"能跑通",要拿**数值与参考实现逐位对齐**;AUROC=0.5 是"全饱和/常数输出"的典型信号。

---
*生成脚本:[`scripts/run_deploy.py`](../scripts/run_deploy.py)。导出产物在 `results/deploy/weights/`(ONNX + Torch,未入库)。*

"""
Week 1 MVP: 在 MVTec AD 上用 PaDiM 跑通无监督缺陷检测 (CPU 友好)。
- 模型: PaDiM (无需反向传播训练, 一次特征提取即可拟合, 适合 CPU)
- 品类: metal_nut (金属螺母, 贴合机械加工件场景)
- 产出: Image/Pixel AUROC 指标 + 缺陷定位热力图 (存到 results/)

数据集: 手动放在 datasets/MVTecAD/metal_nut/ (anomalib 自带下载链接已失效, 见 README)。

注意: 必须用 pandas<3。anomalib 2.5 用 Split 枚举过滤 DataFrame,
pandas 3.0 改了字符串枚举的比较行为, 会导致样本被过滤成 0 (num_samples=0)。
"""
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.models import Padim
from anomalib.engine import Engine
from anomalib.visualization import ImageVisualizer

# 项目根目录 (scripts/ 的上一级)
ROOT = Path(__file__).resolve().parent.parent
CATEGORY = "metal_nut"  # 想换品类改这里: bottle / screw / metal_nut ...

# 1) 数据: 只用 good 样本训练, 测试集含正常+异常
datamodule = MVTecAD(
    root=str(ROOT / "datasets" / "MVTecAD"),
    category=CATEGORY,
    train_batch_size=8,    # CPU 上调小, 防止内存吃紧
    eval_batch_size=8,
    num_workers=0,         # Windows 上设 0, 避免多进程 dataloader 报错
)

# 2) 模型: PaDiM (CPU 友好, 无梯度训练)
model = Padim()

# 3) 引擎: 加可视化回调, 自动把热力图写到 results/
engine = Engine(
    callbacks=[ImageVisualizer()],
    default_root_dir=str(ROOT / "results"),
    accelerator="cpu",
    devices=1,
    enable_progress_bar=False,  # 关掉 rich 进度条: Windows GBK 控制台渲染 '•' 会崩
)

if __name__ == "__main__":
    print(f"[1/3] 拟合 PaDiM (品类={CATEGORY}, 仅用 good 样本)...")
    engine.fit(datamodule=datamodule, model=model)

    print("[2/3] 在测试集上评测 (Image/Pixel AUROC)...")
    engine.test(datamodule=datamodule, model=model)

    print("[3/3] 生成缺陷热力图...")
    engine.predict(datamodule=datamodule, model=model)

    print("\n完成! 热力图在 results/ 目录下 (找 images/ 子文件夹)。")

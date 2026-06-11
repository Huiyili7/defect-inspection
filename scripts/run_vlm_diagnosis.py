"""
Week 5: VLM 诊断层 —— 检测→定位→裁剪→VLM 自然语言诊断。

管线:
  1. 用 W4 导出的 PaDiM (TorchInferencer) 对输入图打异常分 + 出热力图
  2. 从热力图定位高响应区, 裁剪缺陷局部
  3. 把裁剪图 + "缺陷知识库"(RAG 味道) 喂给 Qwen2.5-VL (DashScope qwen-vl-max)
  4. 输出: 缺陷类型 + 机械成因 + 处理建议 (自然语言)

VLM 走 DashScope 的 OpenAI 兼容接口。未设 DASHSCOPE_API_KEY 时用 mock 兜底,
流程照样跑通; 设了 key 同一脚本直接出真实诊断。

运行:
  set TRUST_REMOTE_CODE=1 && set DASHSCOPE_API_KEY=sk-xxx && set PYTHONUTF8=1
  python scripts/run_vlm_diagnosis.py [可选:图片路径]
"""
import base64
import io
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
PT_PATH = ROOT / "results" / "deploy" / "weights" / "torch" / "model.pt"
DATA = ROOT / "datasets" / "MVTecAD" / "metal_nut"

# 缺陷知识库 (RAG: 作为 VLM 的领域参考, 让诊断有依据而非凭空生成)
DEFECT_KB = {
    "bent (弯曲)": "局部塑性变形, 常因冲压成形受力不均、搬运磕碰或装夹施力不当。",
    "scratch (划痕)": "表面细线状损伤, 低对比度, 常因加工/转运摩擦、刀具或夹具刮擦。",
    "color (色差/污染)": "局部变色或附着物, 常因热处理不均、油污残留、氧化或电镀异常。",
    "flip (翻面/装反)": "零件整体朝向错误, 属上料/装夹环节失误, 非表面缺陷。",
}

SYSTEM_PROMPT = (
    "你是资深工业质检与机械加工专家。以下是金属螺母(metal nut)常见缺陷参考知识库:\n"
    + "\n".join(f"- {k}: {v}" for k, v in DEFECT_KB.items())
    + "\n请结合图像与上述知识, 专业、简洁地给出诊断。不确定时如实说明, 不要编造。"
)

USER_PROMPT_TMPL = (
    "这是一张金属螺母图像, 红框是无监督异常检测模型定位的缺陷区域 (整图异常分 {score:.2f}, 越高越异常)。\n"
    "请**结合整体形状轮廓**(判断 bent/flip 等几何类缺陷需要看整体)与红框内的局部细节, 分点输出:\n"
    "1) 缺陷类型 (从知识库中选最匹配的, 或说明是其他)\n"
    "2) 可能成因 (从机械加工/工艺角度)\n"
    "3) 处理 / 改善建议\n"
    "每点一两句, 专业克制。"
)


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def detect_and_localize(image_path: str):
    """返回 (异常分, 缺陷裁剪图, bbox, 原图)。"""
    from anomalib.deploy import TorchInferencer

    inf = TorchInferencer(path=str(PT_PATH), device="cpu")
    res = inf.predict(image=image_path)
    score = float(np.asarray(res.pred_score).reshape(-1)[0])

    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    amap = np.asarray(res.anomaly_map).squeeze().astype(np.float32)
    amap_resized = np.asarray(Image.fromarray(amap).resize((W, H), Image.BILINEAR))

    # 取最热的 top 1% 像素定位峰值区 (0.5*max 太松会圈住弥散背景, 等于不裁)
    thr = float(np.quantile(amap_resized, 0.99))
    ys, xs = np.where(amap_resized >= thr)
    if len(xs) > 10:
        pad = 25
        x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad)
        y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad)
        bbox = (int(x0), int(y0), int(x1), int(y1))
        crop = img.crop(bbox)
    else:
        bbox, crop = None, img  # 无明显热点 -> 用整图

    # 整图 + 红框: 既给 VLM 定位, 又保留整体几何 (bent/flip 类缺陷必须看全局)
    annotated = img.copy()
    if bbox is not None:
        d = ImageDraw.Draw(annotated)
        d.rectangle(bbox, outline=(255, 0, 0), width=6)
    return score, crop, annotated, bbox, img


def mock_diagnosis(score: float) -> str:
    return (
        "[MOCK 占位 —— 未检测到 DASHSCOPE_API_KEY, 以下为示意, 设置 key 后即为真实 VLM 输出]\n"
        f"1) 缺陷类型: 疑似表面缺陷 (异常分 {score:.2f})\n"
        "2) 可能成因: 加工或转运过程中的局部损伤/污染 (示意)\n"
        "3) 处理建议: 复检该工位夹具与转运路径 (示意)"
    )


def diagnose_with_vlm(crop: Image.Image, score: float) -> str:
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        return mock_diagnosis(score)
    from openai import OpenAI

    client = OpenAI(api_key=key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    b64 = pil_to_b64(crop)
    resp = client.chat.completions.create(
        model="qwen-vl-max",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": USER_PROMPT_TMPL.format(score=score)},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    # 默认用一张 bent 缺陷图
    img_path = sys.argv[1] if len(sys.argv) > 1 else str(next((DATA / "test" / "bent").glob("*.png")))

    print(f"输入图片: {img_path}")
    print("[1/3] 检测 + 定位...")
    score, crop, annotated, bbox, img = detect_and_localize(img_path)
    verdict = "异常 (NG)" if score > 0.5 else "正常 (OK)"
    print(f"   判定: {verdict} | 异常分: {score:.3f} | 缺陷框: {bbox}")

    # 存裁剪图 + 标注整图供查看
    out_dir = ROOT / "results" / "vlm"
    out_dir.mkdir(parents=True, exist_ok=True)
    crop.save(out_dir / "last_crop.png")
    annotated.save(out_dir / "last_annotated.png")
    print(f"   缺陷裁剪图: {out_dir / 'last_crop.png'}; 标注整图: {out_dir / 'last_annotated.png'}")

    print("[2/3] 调用 VLM 诊断 (qwen-vl-max)...")
    diagnosis = diagnose_with_vlm(annotated, score)

    print("[3/3] 诊断结果:\n")
    print("=" * 50)
    print(diagnosis)
    print("=" * 50)

    backend = "qwen-vl-max" if os.environ.get("DASHSCOPE_API_KEY") else "MOCK (未设 key)"
    (out_dir / "diagnosis_demo.md").write_text(
        f"# VLM 诊断 demo\n\n- 输入: `{Path(img_path).parent.name}/{Path(img_path).name}`\n"
        f"- 判定: {verdict} | 异常分: {score:.3f} | 缺陷框: {bbox}\n- 后端: {backend}\n\n"
        f"## 诊断\n\n{diagnosis}\n", encoding="utf-8")
    print(f"\n已写入 {out_dir / 'diagnosis_demo.md'}")

import torch
import re
import random
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info


# ==========================================
# 配置与初始化
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


def run_task_2_scienceqa():
    print("\n" + "="*60)
    print("任务二：ScienceQA 多模态科学问答 (Qwen2-VL-7B)")
    print("="*60)

    # ------------------------------------------------------------
    # Step 1: 加载数据集
    # ------------------------------------------------------------
    print("[1/5] 正在加载 ScienceQA 数据集...")
    dataset = load_dataset("derek-thomas/ScienceQA", split="test")

    print("[2/5] 正在筛选包含图片的样本...")
    dataset_with_img = dataset.filter(lambda x: x['image'] is not None)

    # 随机 50 条
    sample_size = 50
    if len(dataset_with_img) < 50:
        sample_size = len(dataset_with_img)

    test_samples = dataset_with_img.shuffle(seed=42).select(range(sample_size))
    print(f"已抽取 {sample_size} 条带图片样本。")

    # ------------------------------------------------------------
    # Step 2: 加载模型（4-bit）
    # ------------------------------------------------------------
    model_path = "/root/autodl-tmp/models/Qwen2-VL-7B-Instruct"
    print(f"[3/5] 正在加载 4-bit 模型：{model_path}")

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    try:
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quant_cfg,
            trust_remote_code=True
        )

        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )
    except Exception as e:
        print(f"模型加载失败：{e}")
        return

    # ------------------------------------------------------------
    # Step 3: 推理
    # ------------------------------------------------------------
    print("[4/5] 开始推理 ...")
    correct = 0
    total = len(test_samples)

    options_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

    for idx, sample in enumerate(test_samples):
        question = sample["question"]
        choices = sample["choices"]
        correct_idx = sample["answer"]
        image = sample["image"]

        gt = options_map[correct_idx]

        # 1. 构造 prompt
        choices_str = "\n".join([f"{options_map[i]}. {c}" for i, c in enumerate(choices)])
        prompt = (
            f"Question: {question}\n"
            f"Options:\n{choices_str}\n"
            f"Answer with the option letter (A/B/C/D)."
        )

        # 2. 构建文本输入
        text = prompt  # 直接将 prompt 作为文本输入

        # 3. 确保图像格式是 PIL，并且尺寸符合模型要求
        if isinstance(image, Image.Image):
            image = image.convert("RGB")  # 确保图像是 RGB 格式
        else:
            print(f"警告：图像格式不正确，正在跳过样本 {idx+1}")
            continue

        # 4. 使用 processor 处理文本和图像
        inputs = processor(
            text=[text],  # 单个文本输入
            images=[image],  # 图像输入必须是列表格式
            return_tensors="pt"
        ).to(model.device)

        # 5. 生成结果
        output_ids = model.generate(**inputs, max_new_tokens=64)

        # 6. 解码生成的 token
        response = processor.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0]

        # 7. 正则提取预测答案
        match = re.search(r"\b([A-E])\b", response)
        pred = match.group(1) if match else "None"

        if pred == gt:
            correct += 1

        print(f"Sample {idx+1:02d} | GT: {gt} | Pred: {pred} | {'correct' if pred == gt else 'wrong'}")

    # ------------------------------------------------------------
    # Step 4: 结果统计
    # ------------------------------------------------------------
    acc = correct / total
    print("-" * 60)
    print("[5/5] 评测结束")
    print(f"最终准确率: {acc:.2%} ({correct}/{total})")
    print("-" * 60)


if __name__ == "__main__":
    run_task_2_scienceqa()

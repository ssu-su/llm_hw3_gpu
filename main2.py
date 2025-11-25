import torch
import re
import random
import numpy as np
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
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

# 设置随机种子，确保每次抽取的 50 条数据一致
set_seed(42)

def run_task_2_scienceqa():
    print("\n" + "="*50)
    print("任务二：ScienceQA 多模态科学问答评测 (Qwen2-VL-7B)")
    print("="*50)

    # ----------------------------------------------------------------
    # Step 1: 数据准备 (Pipeline: Load -> Filter -> Sample)
    # ----------------------------------------------------------------
    print("[1/5] 正在加载 ScienceQA 数据集...")
    try:
        # 加载测试集 [cite: 9]
        dataset = load_dataset("derek-thomas/ScienceQA", split="test")
    except Exception as e:
        print(f"数据加载失败，请检查网络连接: {e}")
        return

    # 筛选包含图片的数据 (image 字段不为 None) 
    print("[2/5] 正在筛选包含图片的样本...")
    dataset_with_img = dataset.filter(lambda x: x['image'] is not None)
    
    # 随机抽取 50 条数据 [cite: 11]
    sample_size = 50
    if len(dataset_with_img) < sample_size:
        print(f"警告：带图片样本不足 {sample_size} 条，使用全部 {len(dataset_with_img)} 条。")
        sample_size = len(dataset_with_img)
    
    indices = random.sample(range(len(dataset_with_img)), sample_size)
    test_samples = dataset_with_img.select(indices)
    print(f"已抽取 {len(test_samples)} 条带图片的测试样本。")

    # ----------------------------------------------------------------
    # Step 2: 模型加载 (Pipeline: Load Model & Processor in 4-bit)
    # ----------------------------------------------------------------
    model_name = "Qwen/Qwen2-VL-7B-Instruct" # [cite: 12]
    print(f"[3/5] 正在加载模型: {model_name} (4-bit 量化)...")

    # 配置 4-bit 量化参数 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # 加载多模态模型
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype="auto",
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        # 加载对应的处理器 (处理图片和文本)
        # min_pixels 和 max_pixels 限制图片分辨率以控制显存
        processor = AutoProcessor.from_pretrained(
            model_name, 
            min_pixels=256*28*28, 
            max_pixels=1280*28*28
        )
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # ----------------------------------------------------------------
    # Step 3: 推理循环 (Pipeline: Preprocess -> Generate -> Extract)
    # ----------------------------------------------------------------
    correct_count = 0
    total_count = len(test_samples)
    
    # 选项索引映射 (0->A, 1->B ...)
    options_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

    print(f"[4/5] 开始推理 (共 {total_count} 条)...")
    print("-" * 60)

    for i, sample in enumerate(test_samples):
        # 1. 准备数据
        question = sample['question']
        choices = sample['choices'] # 选项列表
        answer_idx = sample['answer'] # 正确选项的索引 (int)
        image = sample['image'] # PIL Image 对象
        
        ground_truth = options_map.get(answer_idx, "Unknown")
        
        # 2. 构建 Prompt
        # 格式化选项字符串
        choices_str = "\n".join([f"{options_map[idx]}. {c}" for idx, c in enumerate(choices)])
        
        prompt_text = (
            f"Question: {question}\n"
            f"Options:\n{choices_str}\n"
            "Answer with the option letter directly (e.g., A, B, C, D)."
        )

        # Qwen2-VL 的消息格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # 3. 处理输入 (Image + Text)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # 4. 模型生成
        # max_new_tokens 设置较小，因为只需要输出选项字母
        generated_ids = model.generate(**inputs, max_new_tokens=64)
        
        # 移除输入部分的 tokens，只保留生成的 tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # 5. 提取答案 (Post-processing)
        # 使用正则提取第一个出现的 A-E 字母
        match = re.search(r'([A-E])', response.strip())
        prediction = match.group(1) if match else "None"

        # 6. 对比结果
        is_correct = (prediction == ground_truth)
        if is_correct:
            correct_count += 1
        
        # 打印日志
        print(f"Sample {i+1:02d} | GT: {ground_truth} | Pred: {prediction} | {'✅ Correct' if is_correct else '❌ Wrong'}")
        # 可选：打印更详细的信息以便调试
        # print(f"  Q: {question[:50]}...") 

    # ----------------------------------------------------------------
    # Step 4: 结果统计
    # ----------------------------------------------------------------
    accuracy = correct_count / total_count
    print("-" * 60)
    print(f"[5/5] 评测结束")
    print(f"最终准确率 (Accuracy): {accuracy:.2%} ({correct_count}/{total_count})")
    print("-" * 60)

if __name__ == "__main__":
    run_task_2_scienceqa()
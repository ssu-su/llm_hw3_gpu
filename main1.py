import torch
import re
import random
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# 设置随机种子以保证结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==========================================
# 任务一: LLM 数学推理评测 (GSM8K)
# ==========================================
def run_task_1_gsm8k():
    print("\n" + "="*50)
    print("开始任务一：GSM8K 数学推理评测")
    print("="*50)

    # 1. 加载数据集 [cite: 5]
    print("正在加载 GSM8K 数据集...")
    try:
        dataset = load_dataset("openai/gsm8k", "main", split="test")
    except Exception as e:
        print(f"加载数据集失败，请检查网络: {e}")
        return

    # 2. 随机抽取 50 条 [cite: 6]
    indices = random.sample(range(len(dataset)), 50)
    test_samples = dataset.select(indices)
    
    # 3. 加载模型 (推荐 Qwen2.5-7B-Instruct 用于纯文本任务) [cite: 7]
    # 使用 4-bit 量化以节省显存 
    model_name = "Qwen/Qwen2.5-7B-Instruct" 
    print(f"正在加载模型: {model_name} (4-bit)...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )

    correct_count = 0
    total_count = len(test_samples)

    print("开始推理...")
    for i, sample in enumerate(test_samples):
        question = sample['question']
        ground_truth_str = sample['answer'] # 格式通常包含推理过程和 #### 答案
        
        # 提取 Ground Truth 数值 (通常在 #### 之后)
        ground_truth_val = ground_truth_str.split("####")[-1].strip()
        
        # 构建 Prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Solve the math problem step by step. Finally, output the answer strictly in the format: #### Number"},
            {"role": "user", "content": question}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 生成回答
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.01 # 降低随机性
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # 提取模型预测的答案
        # 尝试寻找 #### 后的数字，如果没有则尝试寻找最后一个数字
        pred_match = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', response)
        if pred_match:
            pred_val = pred_match.group(1).replace(',', '')
        else:
            # Fallback: 找最后一个数字
            all_nums = re.findall(r'-?[\d,]+(?:\.\d+)?', response)
            pred_val = all_nums[-1].replace(',', '') if all_nums else "Error"

        # 对比 (简单的字符串对比，也可以转float对比)
        is_correct = False
        try:
            if float(pred_val) == float(ground_truth_val):
                is_correct = True
        except:
            pass
            
        if is_correct:
            correct_count += 1

        print(f"[{i+1}/{total_count}] | GT: {ground_truth_val} | Pred: {pred_val} | {'Correct' if is_correct else 'Wrong'}")

    accuracy = correct_count / total_count
    print(f"\n任务一 GSM8K 准确率: {accuracy:.2%} ({correct_count}/{total_count})")
    
    # 清理显存
    del model
    del tokenizer
    torch.cuda.empty_cache()

# ==========================================
# 任务二: MLLM 多模态科学问答 (ScienceQA)
# ==========================================
def run_task_2_scienceqa():
    print("\n" + "="*50)
    print("开始任务二：ScienceQA 多模态问答")
    print("="*50)

    # 1. 加载数据集 [cite: 9]
    print("正在加载 ScienceQA 数据集...")
    dataset = load_dataset("derek-thomas/ScienceQA", split="test")

    # 2. 筛选包含图片的数据 [cite: 10]
    # ScienceQA 的 image 字段如果不是 None 则是 PIL Image 对象
    dataset_with_img = dataset.filter(lambda x: x['image'] is not None)
    
    # 3. 随机抽取 50 条 [cite: 11]
    # 注意：如果筛选后数量不足50，则取全部
    sample_size = min(50, len(dataset_with_img))
    indices = random.sample(range(len(dataset_with_img)), sample_size)
    test_samples = dataset_with_img.select(indices)

    # 4. 加载模型 (指定 Qwen2-VL-7B-Instruct) [cite: 12]
    model_name = "/root/autodl-tmp/models/Qwen2-VL-7B-Instruct"
    print(f"正在加载模型: {model_name} (4-bit)...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # Qwen2-VL 需要特定的 processor
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype="auto",
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    
    # 这里的 min_pixels 和 max_pixels 可以根据显存调整
    processor = AutoProcessor.from_pretrained(model_name, min_pixels=256*28*28, max_pixels=1280*28*28)

    correct_count = 0
    total_count = len(test_samples)

    print("开始推理...")
    for i, sample in enumerate(test_samples):
        question = sample['question']
        choices = sample['choices'] # List of strings
        answer_idx = sample['answer'] # Int: 0, 1, 2...
        image = sample['image'] # PIL Image

        # 将数字索引转换为选项字母 (0->A, 1->B...)
        options_map = {k: v for k, v in enumerate(['A', 'B', 'C', 'D', 'E'])}
        ground_truth_opt = options_map[answer_idx]

        # 格式化选项文本
        choices_str = "\n".join([f"{options_map[idx]}. {choice}" for idx, choice in enumerate(choices)])
        
        # 构建 Prompt [cite: 13]
        prompt_text = f"Question: {question}\nOptions:\n{choices_str}\nAnswer with the option letter directly."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # 准备输入
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

        # 生成
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # 提取答案 (通常模型会直接输出 "A" 或者 "The answer is A")
        # 简单提取第一个出现的 A-E 字母
        match = re.search(r'([A-E])', response)
        pred_opt = match.group(1) if match else "None"

        is_correct = (pred_opt == ground_truth_opt)
        if is_correct:
            correct_count += 1
            
        print(f"[{i+1}/{total_count}] | GT: {ground_truth_opt} | Pred: {pred_opt} | {'Correct' if is_correct else 'Wrong'}")

    accuracy = correct_count / total_count
    print(f"\n任务二 ScienceQA 准确率: {accuracy:.2%} ({correct_count}/{total_count})")

if __name__ == "__main__":
    # 按顺序执行任务
    run_task_1_gsm8k()
    run_task_2_scienceqa()
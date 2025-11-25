import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import re
import random
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# 设置随机种子以保证可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

class GSM8KEvaluator:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", quantize_4bit=True):
        """
        初始化GSM8K评测器
        
        Args:
            model_name: 模型名称
            quantize_4bit: 是否使用4-bit量化
        """
        self.model_name = model_name
        self.quantize_4bit = quantize_4bit
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"正在加载模型: {model_name}")
        print(f"设备: {self.device}")
        print(f"4-bit量化: {quantize_4bit}")
        
        # 配置4-bit量化
        if quantize_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            quantization_config = None
        
        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # 添加pad_token如果不存在
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("模型加载完成!")
    
    def extract_answer(self, text):
        """
        从模型输出中提取最终答案
        
        Args:
            text: 模型生成的文本
            
        Returns:
            extracted_answer: 提取的数字答案
        """
        # 多种模式匹配最终答案
        patterns = [
            r'####\s*(\-?\d+(?:\.\d+)?)',  # #### 123
            r'答案\s*[:：]\s*(\-?\d+(?:\.\d+)?)',  # 答案: 123
            r'答案是\s*(\-?\d+(?:\.\d+)?)',  # 答案是123
            r'[Tt]he answer is\s*(\-?\d+(?:\.\d+)?)',  # The answer is 123
            r'[Ff]inal answer\s*[:：]?\s*(\-?\d+(?:\.\d+)?)',  # Final answer: 123
            r'(\-?\d+(?:\.\d+)?)\s*$'  # 行末的数字
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                # 返回最后一个匹配，因为通常最终答案在最后
                return matches[-1]
        
        # 如果没有匹配到特定模式，尝试提取所有数字并返回最后一个
        numbers = re.findall(r'\-?\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]
        
        return None
    
    def extract_ground_truth(self, answer_text):
        """
        从数据集的answer字段提取真实答案
        
        Args:
            answer_text: 数据集的answer字段文本
            
        Returns:
            ground_truth: 真实答案
        """
        # GSM8K数据集的答案格式通常是：推理过程 #### 数字答案
        match = re.search(r'####\s*(\-?\d+(?:\.\d+)?)', answer_text)
        if match:
            return match.group(1)
        return None
    
    def generate_response(self, question, max_length=1024):
        """
        生成模型的回答
        
        Args:
            question: 数学问题
            max_length: 最大生成长度
            
        Returns:
            response: 模型生成的完整回答
        """
        # 构建prompt
        prompt = f"请解决以下数学问题，并给出详细的推理过程。在最后一行用####标注最终答案。\n\n问题：{question}\n\n解答："
        
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成回答
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_return_sequences=1,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码生成文本
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 只返回生成的部分（去掉prompt）
        generated_text = response[len(prompt):].strip()
        
        return generated_text
    
    def evaluate(self, num_samples=50):
        """
        在GSM8K测试集上进行评测
        
        Args:
            num_samples: 评测样本数量
            
        Returns:
            results: 评测结果字典
        """
        print(f"正在加载GSM8K测试集并随机抽取{num_samples}条数据...")
        
        # 加载数据集
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        
        # 随机抽样
        if len(dataset) > num_samples:
            indices = random.sample(range(len(dataset)), num_samples)
            test_samples = dataset.select(indices)
        else:
            test_samples = dataset
        
        print(f"成功加载{len(test_samples)}个测试样本")
        
        results = {
            'total': len(test_samples),
            'correct': 0,
            'accuracy': 0.0,
            'details': []
        }
        
        print("开始评测...")
        for i, sample in enumerate(tqdm(test_samples, desc="评测进度")):
            question = sample['question']
            ground_truth_answer = self.extract_ground_truth(sample['answer'])
            
            if ground_truth_answer is None:
                print(f"警告: 无法从样本{i}提取真实答案")
                continue
            
            try:
                # 生成模型回答
                model_response = self.generate_response(question)
                
                # 提取模型答案
                model_answer = self.extract_answer(model_response)
                
                # 检查答案是否正确
                is_correct = False
                if model_answer is not None:
                    # 尝试将答案转换为浮点数进行比较
                    try:
                        model_float = float(model_answer)
                        truth_float = float(ground_truth_answer)
                        # 允许小的浮点数误差
                        is_correct = abs(model_float - truth_float) < 1e-6
                    except ValueError:
                        # 如果转换失败，进行字符串比较
                        is_correct = (model_answer.strip() == ground_truth_answer.strip())
                
                if is_correct:
                    results['correct'] += 1
                
                # 保存详细信息
                results['details'].append({
                    'question': question,
                    'ground_truth': ground_truth_answer,
                    'model_response': model_response,
                    'model_answer': model_answer,
                    'is_correct': is_correct
                })
                
            except Exception as e:
                print(f"处理样本{i}时出错: {e}")
                results['details'].append({
                    'question': question,
                    'ground_truth': ground_truth_answer,
                    'model_response': f"Error: {e}",
                    'model_answer': None,
                    'is_correct': False
                })
        
        # 计算准确率
        results['accuracy'] = results['correct'] / results['total']
        
        return results
    
    def print_results(self, results):
        """
        打印评测结果
        
        Args:
            results: 评测结果字典
        """
        print("\n" + "="*80)
        print("GSM8K数学推理评测结果")
        print("="*80)
        print(f"模型: {self.model_name}")
        print(f"测试样本数: {results['total']}")
        print(f"正确回答数: {results['correct']}")
        print(f"准确率: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print("="*80)
        
        # 打印前5个样本的详细信息
        print("\n前5个样本的详细结果:")
        print("-"*80)
        
        for i, detail in enumerate(results['details'][:5]):
            print(f"\n样本 {i+1}:")
            print(f"问题: {detail['question']}")
            print(f"真实答案: {detail['ground_truth']}")
            print(f"模型答案: {detail['model_answer']}")
            print(f"模型回答: {detail['model_response'][:200]}...")
            print(f"是否正确: {'✓' if detail['is_correct'] else '✗'}")
            print("-"*80)

def main():
    """
    主函数：执行GSM8K评测任务
    """
    # 模型选择
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    # 初始化评测器
    evaluator = GSM8KEvaluator(model_name=model_name, quantize_4bit=True)
    
    # 模型结构说明
    print("\n" + "="*80)
    print("模型结构说明 (基于Qwen2.5-7B-Instruct原论文)")
    print("="*80)
    print("1. 架构: Transformer解码器架构")
    print("2. 参数量: 7B (70亿参数)")
    print("3. 上下文长度: 32K tokens")
    print("4. 注意力机制: Group Query Attention (GQA)")
    print("5. 位置编码: Rotary Position Embedding (RoPE)")
    print("6. 激活函数: SwiGLU")
    print("7. 归一化: RMSNorm")
    print("8. 词汇表大小: 152,064")
    print("="*80)
    
    # 进行评测
    results = evaluator.evaluate(num_samples=50)
    
    # 打印结果
    evaluator.print_results(results)
    
    # 保存详细结果到文件
    import json
    with open('gsm8k_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到: gsm8k_results.json")

if __name__ == "__main__":
    main()
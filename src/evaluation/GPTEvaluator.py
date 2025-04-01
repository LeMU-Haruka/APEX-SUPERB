from src.evaluation import Evaluator
from openai import OpenAI

class GPTEvaluator(Evaluator):

    def __init__(self):
        self.client = OpenAI()


    def __call_api(self, message):
        completion = self.client.chat.completions.create(
            model="gpt-4",  # 改为正式模型名 "gpt-4"
            messages=message,
            max_tokens=50,  # 控制生成的最大字数
            temperature=0.7,  # 控制创造性，范围是 0 到 1，值越高，生成越随机
            top_p=0.9,  # 通过概率质量选择生成的词汇，top_p 越高选择越多样
            frequency_penalty=0.2,  # 控制重复度，越高越少重复
            presence_penalty=0.3  # 鼓励生成新的词汇
        )

from openai import OpenAI

from src.evaluation.evaluators.evaluator import Evaluator

class GPTClient(Evaluator):

    def __init__(self):
        self.client = OpenAI()


    def __build_messages(self, prompt: str) -> list:
        """
        Build messages formatted for OpenAI chat completion API.
        
        Args:
            prompt (str): The input task prompt.

        Returns:
            list: A list of message dictionaries formatted for the API.
        """
        messages = [
            {"role": "system", "content": "You are a professional assistant for evaluating large model generations. Please carefully analyze and respond based on the given prompt."},
            {"role": "user", "content": prompt}
        ]
        return messages

    def __call_api(self, prompt):
        message = self.__build_messages(prompt)
        completion = self.client.chat.completions.create(
            model="gpt-4",  # 改为正式模型名 "gpt-4"
            messages=message,
            max_tokens=200,  # 控制生成的最大字数
            temperature=0.5,  # 控制创造性，范围是 0 到 1，值越高，生成越随机
            top_p=0.9,  # 通过概率质量选择生成的词汇，top_p 越高选择越多样
            frequency_penalty=0,  # 控制重复度，越高越少重复
            presence_penalty=0,  # 鼓励生成新的词汇
            seed=42
        )

    def generate_response(self, prompt):
        response = self.__call_api(prompt)
        return response
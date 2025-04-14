
import json
from openai import OpenAI

from src.evaluation.evaluators.evaluator import Evaluator

class VllmClient:

    def __init__(self):
        openai_api_key = "EMPTY"
        openai_api_base = "http://127.0.0.1:8000/v1"
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
    
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
        messages = self.__build_messages(prompt)

        completion = self.client.chat.completions.create(
            model="/userhome/models/llama-2-7b-chat-hf",  # 改为正式模型名 "gpt-4"
            # model='/userhome/models/Llama-3.1-8B-Instruct',
            messages=messages,
            max_tokens=500,  # 控制生成的最大字数
            temperature=0.5,  # 控制创造性，范围是 0 到 1，值越高，生成越随机
            top_p=0.9,  # 通过概率质量选择生成的词汇，top_p 越高选择越多样
            # frequency_penalty=0,  # 控制重复度，越高越少重复
            # presence_penalty=0,  # 鼓励生成新的词汇
            seed=42
        )
        return completion.choices[0].message.content

    def generate_response(self, prompt):
        response = self.__call_api(prompt)
        return response
    
    # def evaluate(self, data):
    #     preds = []
    #     targets = []
    #     origin_preds = []
    #     for item in data:
    #         if self.is_align:
    #             pred = self.align_text(item)
    #             origin_preds.append(item['pred'])
    #         else:
    #             pred = item['pred']
    #         preds.append(pred)
    #         targets.append(item['target'])
    #     if self.is_align:
    #         json_str = json.dumps({'origin': origin_preds, 'pred': preds, 'target': targets}, indent=4)
    #         with open(f'./cache/{self.task}_aligned.json', 'w') as f:
    #             f.write(json_str)
    #     score = self.metric(preds, targets)
    #     return score
import os
import random
import time
from google import genai
from google.genai import types


class GeminiClient:

    def __init__(self):
        self.client = genai.Client(api_key='API KEY')
        

    def print_support_models(self):
        for m in self.client.models.list():
                print(m)


    def __call_api(self, message):
        generation_config = types.GenerateContentConfig(
            temperature=0.5,  # 从kwargs中获取，如果没有则使用默认值
            top_k=40,
            top_p=0.9,
            max_output_tokens=500,
            # candidate_count=kwargs.get("candidate_count", 1),  # 如果需要，也可以设置
            # stop_sequences=kwargs.get("stop_sequences", None)   # 如果需要
        )
        response = self.client.models.generate_content(
            # model='gemini-2.5-pro-preview-03-25',
            model='gemini-2.5-flash-preview-04-17',
            contents=[message],
            config=generation_config
            )
        response = response.text
        return response


    def generate_response(self, prompt):
        try:
            response = self.__call_api(prompt)
        except Exception as e:
            print(f"Error occurred: {e}")
            print("Retrying...")
            time.sleep(10)
            response = self.__call_api(prompt)
        return response
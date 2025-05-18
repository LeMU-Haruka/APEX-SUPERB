import os
import random
import time
from google import genai
from google.genai import types

# 配置代理
# os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890"

class GeminiClient:

    def __init__(self):
        # os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890"
        self.client = genai.Client(api_key='API KEY')
        

    def print_support_models(self):
        for m in self.client.models.list():
                print(m)


    def __call_api(self, message):
        # delay to avoid to meet rate limit
        # time.sleep(random.uniform(1.5, 3.5))
        # generation_config = types.GenerateContentConfig(
        #     temperature=0.5,  # 从kwargs中获取，如果没有则使用默认值
        #     # top_k=40,
        #     # top_p=0.9,
        #     max_output_tokens=500,
        #     # candidate_count=kwargs.get("candidate_count", 1),  # 如果需要，也可以设置
        #     # stop_sequences=kwargs.get("stop_sequences", None)   # 如果需要
        # )
        response = self.client.models.generate_content(
            # model='gemini-2.5-pro-preview-03-25',
            model='gemini-2.5-flash-preview-04-17',
            contents=[message],
            # config=generation_config
            )
        response = response.text
        # match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL | re.IGNORECASE)

        # if match:
        #     json_str = match.group(1).strip() # Extract the captured group (the {...} part)
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
    
    # def extract_json(self, text: str) -> dict | None:
    #     """
    #     从可能包含 Markdown 代码块或前缀/后缀文本的字符串中提取第一个有效的 JSON 对象。

    #     Args:
    #         text: 包含潜在 JSON 的原始字符串。

    #     Returns:
    #         解析后的 Python 字典，如果找不到或无法解析则返回 None。
    #     """
    #     if not text:
    #         return None

    #     # 1. 优先查找 Markdown 代码块 (```json ... ``` or ``` ... ```)
    #     #    使用非贪婪匹配 .*? 来获取第一个完整的 {...} 对
    #     match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL | re.IGNORECASE)
    #     if match:
    #         json_str = match.group(1).strip()
    #         try:
    #             return json.loads(json_str)
    #         except json.JSONDecodeError:
    #             # 如果 Markdown 块内的内容不是有效 JSON，则继续尝试下面的方法
    #             pass

    #     # 2. 如果没找到 Markdown 块，或者块内 JSON 无效，
    #     #    则尝试查找第一个 '{' 和最后一个 '}' 之间的内容
    #     start_index = text.find('{')
    #     end_index = text.rfind('}')
    #     if start_index != -1 and end_index != -1 and end_index > start_index:
    #         json_str = text[start_index : end_index + 1].strip()
    #         try:
    #             return json.loads(json_str)
    #         except json.JSONDecodeError:
    #             # 如果这部分也不是有效 JSON，则失败
    #             pass

    #     # 3. 如果以上方法都失败
    #     return None

   

import random
import time
from openai import OpenAI

from src.evaluation.evaluators.evaluator import Evaluator

class GPTClient(Evaluator):

    def __init__(self):
        openai_api_key = "eyJhbGciOiJSUzI1NiIsImtpZCI6IjE5MzQ0ZTY1LWJiYzktNDRkMS1hOWQwLWY5NTdiMDc5YmQwZSIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS92MSJdLCJjbGllbnRfaWQiOiJhcHBfWDh6WTZ2VzJwUTl0UjNkRTduSzFqTDVnSCIsImV4cCI6MTc0NDk2MzQ0NiwiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS9hdXRoIjp7InVzZXJfaWQiOiJ1c2VyLUV4TWt5aU1CcDdqT2puM0RhbjQyaVFLeCJ9LCJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJqaW5ncmFueGllQGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlfSwiaWF0IjoxNzQ0MDk5NDQ1LCJpc3MiOiJodHRwczovL2F1dGgub3BlbmFpLmNvbSIsImp0aSI6IjM5MGUwYzY1LWY5NjktNGExYi04M2U0LTY1MWU0NTc5MDUyNiIsIm5iZiI6MTc0NDA5OTQ0NSwicHdkX2F1dGhfdGltZSI6MTczNjg0NTc5MTU5Niwic2NwIjpbIm9wZW5pZCIsImVtYWlsIiwicHJvZmlsZSIsIm9mZmxpbmVfYWNjZXNzIiwibW9kZWwucmVxdWVzdCIsIm1vZGVsLnJlYWQiLCJvcmdhbml6YXRpb24ucmVhZCIsIm9yZ2FuaXphdGlvbi53cml0ZSJdLCJzZXNzaW9uX2lkIjoiYXV0aHNlc3NfSGtwZ0VBYTBiRk1HTWZUS1VoRlB1MHN3Iiwic3ViIjoiZ29vZ2xlLW9hdXRoMnwxMTczODkwODAxNjI0MzQwMTEwMDcifQ.rc7zIpFb7W9KyOZP7jPrVTP-We8iUxPyX6ze0pI2758HT9ppi-8MxrQGv3413ziSnquFSP5nx7dRRcQdvTsiqhSbzqyPK873R8thZuTQlM185I9k1BxJp7Ku7LUsnvN0ylsQojYSzDNiipy17zIpSKcid_PSasgUkSBad7Z5ivD0nvjMyeLwosfE9FxwEGQY-WJjkUMoGN-EV-5rbF4TxwhPioPgjHLsftNBh-nJPRqLM7p0p_UT2yn93jzSH9nhQeTA2NTRw0B765Y1ajpypVViJu9bSNemzn5gh4ULL-IC7T5ZN4VQRar575DuIsF5M1lDN4K0bqXbq5bDr_uYewZCgC-pyxVMKrDToCqRC2u72db1CzrF9Io7KLszdd8Jjc9cNAmzVtMe7mOPb1NALvqEpOrqrfrPmsULtq_P9pVWQJOnMsEMtJjcqSvwgION16OnHIu6bO4c8z9nY9adFTOyE-UmqrluzvfL8wL2Z5ecrxKPMzW5hOLsHYsAjarMKKxjwvr-NQQQQbhqc3samG0As6wbWp3Xz07EZAoOgDvwKG7H_yBwUWX4X4RHRSi91D7uz2NYb7Ac6BmhRqDgid5sIDdc2BMzYCjo4-oaUbgHkXK5kT38fmkzrlMij0i2aiXD8OYem0p7h1s73tTAYI7fS8R-R88f9yXob2nUD6o"
        openai_api_base = "http://127.0.0.1:5005/v1"
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        # self.client = OpenAI()


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
        time.sleep(random.uniform(1.5, 2.5))  # 随机延迟，避免过快请求
        message = self.__build_messages(prompt)
        completion = self.client.chat.completions.create(
            model="gpt-4o",  # 改为正式模型名 "gpt-4"
            messages=message,
            max_tokens=500,  # 控制生成的最大字数
            temperature=0.5,  # 控制创造性，范围是 0 到 1，值越高，生成越随机
            top_p=0.9,  # 通过概率质量选择生成的词汇，top_p 越高选择越多样
            frequency_penalty=0,  # 控制重复度，越高越少重复
            presence_penalty=0,  # 鼓励生成新的词汇
            seed=42
        )
        return completion.choices[0].message.content

    def generate_response(self, prompt):
        try:
            response = self.__call_api(prompt)
        except Exception as e:
            print(f"Error occurred: {e}")
            print("Retrying...")
            time.sleep(10)
            response = self.__call_api(prompt)
        return response
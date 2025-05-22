import time
from openai import OpenAI

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
            max_tokens=500, 
            temperature=0.5, 
            top_p=0.9, 
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
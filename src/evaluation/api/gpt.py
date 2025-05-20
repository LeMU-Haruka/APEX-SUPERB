
import random
import time
from openai import OpenAI

from src.evaluation.evaluators.evaluator import Evaluator

class GPTClient(Evaluator):

    def __init__(self):
        openai_api_key = "API KEY"
        self.client = OpenAI(
            api_key=openai_api_key,
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
        message = self.__build_messages(prompt)
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=message,
            max_tokens=500, 
            temperature=0.5,
            top_p=0.9, 
            frequency_penalty=0, 
            presence_penalty=0,
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
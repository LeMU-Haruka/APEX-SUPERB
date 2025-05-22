import base64
import time
from src.models.base_model import BaseModel
from src.utils import array_to_audio_bytes
from openai import OpenAI


class GPT4oAudio(BaseModel):
    def __init__(self, llm_path='gpt-4o-audio-preview'):
        openai_api_key = "API KEY"
        self.client = OpenAI(
            api_key=openai_api_key,
        )
        self.llm_name = llm_path

    def __call_api(
        self,
        instruction,
        audio_bytes,
    ):
        response = self.client.chat.completions.create(
            model=self.llm_name,
            modalities=["text"],
            messages=instruction,
            max_tokens=1024,
            temperature=0.7,
        )
        return response

    def prompt_mode(
        self,
        prompt,
        audio,
        sr,
        max_new_tokens=1024,
    ):
        audio_bytes = array_to_audio_bytes(audio, sr)
        encoded_string = base64.b64encode(audio_bytes).decode('utf-8')
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": encoded_string,
                        "format": "wav"
                    }
                }
            ]}
        ]
        try:
            response = self.__call_api(messages, encoded_string)
        except Exception as e:
            print(f"Error occurred: {e}")
            print("Retrying...")
            time.sleep(30)
            try:
                response = self.__call_api(messages, encoded_string)
            except Exception as e:
                return 'failed'
        
        response_text = response.choices[0].message.content
        return response_text

    def chat_mode(
        self,
        audio,
        sr,
        max_new_tokens=1024,
    ):
        audio_bytes = array_to_audio_bytes(audio, sr)
        encoded_string = base64.b64encode(audio_bytes).decode('utf-8')
        messages = [
            {"role": "system", "content": "You are a helpful speech assistant to answer question of user."},
            {"role": "user", "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": encoded_string,
                        "format": "wav"
                    }
                }
            ]}
        ]
        try:
            response = self.__call_api(messages, encoded_string)
        except Exception as e:
            print(f"Error occurred: {e}")
            print("Retrying...")
            time.sleep(30)
            response = self.__call_api(messages, encoded_string)
        response_text = response.choices[0].message.content
        return response_text
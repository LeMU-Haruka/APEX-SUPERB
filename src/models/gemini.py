
import os
import time
from google.genai import types
from google import genai
from src.models.base_model import BaseModel

from src.utils import array_to_audio_bytes




class GeminiAudio(BaseModel):
    def __init__(self, llm_path='gemini-1.5-pro'):
        # os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890"
        self.client  = genai.Client(api_key='API KEY')
        self.api_map = {
            'gemini-1.5-pro': 'gemini-1.5-pro',
            'gemin-2.5-flash': 'gemini-2.5-flash-preview-04-17',
            'gemini-2.5-pro': 'gemini-2.5-pro-preview-05-06',
        }
        self.llm_name = self.api_map[llm_path]

    def __call_api(self, instruction, audio_bytes):
        response = self.client.models.generate_content(
            model=self.llm_name,
            contents=[
                instruction,
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type='audio/mp3',
                ),
            ],
            config=types.GenerateContentConfig(
                max_output_tokens=1024,
                temperature=0.5
            )
        )
        return response
        

    def prompt_mode(self, instruction, audio, sr, max_new_tokens=1024):
        # 为了避免出发每分钟请求上限，sleep
        # temp_wav = './cache/gemini_temp.wav'
        # sf.write(temp_wav, audio, sr)
        # with open(temp_wav, 'rb') as f:
        #     audio_bytes = f.read()
        # time.sleep(random.randint(2, 4))
        audio_bytes = array_to_audio_bytes(audio, sr, fmt='wav')

        try:
            response = self.__call_api(
                instruction,
                audio_bytes
            )
        except Exception as e:
            print(f"Error occurred: {e}")
            print("Retrying...")
            time.sleep(30)
            response = self.__call_api(
                instruction,
                audio_bytes
            )
        
        response_text = response.text
        # print(response_text)
        return response_text


    
    def chat_mode(self, audio, sr, max_new_tokens=1024):
        audio_bytes = array_to_audio_bytes(audio, sr, fmt='wav')
        instruction = 'You are a helpful speech assistant to answer the question of user.'
        try:
            response = self.__call_api(
                instruction,
                audio_bytes
            )
        except Exception as e:
            print(f"Error occurred: {e}")
            print("Retrying...")
            time.sleep(30)
            response = self.__call_api(
                instruction,
                audio_bytes
            )
        
        response_text = response.text
        # print(response_text)
        return response_text

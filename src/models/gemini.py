
import os
import time
from google.genai import types
from google import genai
from src.models.base_model import BaseModel

from src.utils import array_to_audio_bytes




class GeminiAudio(BaseModel):
    def __init__(self, llm_name='gemini-2.0-flash'):
        os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890"
        self.client  = genai.Client(api_key='AIzaSyBkXK1UXbn0BUnWEiRhcdryUrWwOgVp3oc')

    def __call_api(self, instruction, audio_bytes):
        response = self.client.models.generate_content(
            model='gemini-1.5-pro',
            contents=[
                instruction,
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type='audio/mp3',
                ),
            ],
            config=types.GenerateContentConfig(
                max_output_tokens=500,
                temperature=0.5
            )
        )
        return response
        

    def prompt_mode(self, instruction, audio, sr, max_new_tokens=2048):
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


    
    def chat_mode(self, audio, sr, max_new_tokens=2048):
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

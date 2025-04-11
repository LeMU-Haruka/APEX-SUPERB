import os
import time
from google.genai import types
from google import genai
from src.models.base_model import BaseModel
import soundfile as sf




class GeminiAudio(BaseModel):
    def __init__(self, llm_name='gemini-2.0-flash'):
        os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890"
        self.client  = genai.Client(api_key='AIzaSyBkXK1UXbn0BUnWEiRhcdryUrWwOgVp3oc')
        


    def prompt_mode(self, instruction, audio, sr, max_new_tokens=2048):
        # 为了避免出发每分钟请求上限，sleep
        time.sleep(5)
        temp_wav = './cache/temp.wav'
        sf.write(temp_wav, audio, sr)
        with open(temp_wav, 'rb') as f:
            audio_bytes = f.read()

        response = self.client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[
                instruction,
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type='audio/mp3',
                )
            ]
        )
        
        response_text = response.text
        print(response_text)
        return response_text

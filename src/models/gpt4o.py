import base64
from src.models.base_model import BaseModel
from src.utils import array_to_audio_bytes
from openai import OpenAI


class GPT4oAudio(BaseModel):
    def __init__(self, llm_path='gpt-4o-audio-preview'):
        openai_api_key = "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"
        self.client = OpenAI(
            api_key=openai_api_key,
        )

    def prompt_mode(
        self,
        prompt,
        audio,
        sr,
        max_new_tokens=2048,
    ):
        audio_bytes = array_to_audio_bytes(audio, sr)
        encoded_string = base64.b64encode(audio_bytes).decode('utf-8')

        response = self.client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text"],
            messages=[
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
            ],
            max_tokens=500,
            temperature=0.7,
        )
        response_text = response.choices[0].message.content
        return response_text

    def chat_mode(
        self,
        audio,
        sr,
        max_new_tokens=2048,
    ):
        audio_bytes = array_to_audio_bytes(audio, sr)
        encoded_string = base64.b64encode(audio_bytes).decode('utf-8')

        response = self.client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text"],
            messages=[
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
            ],
            max_tokens=500,
            temperature=0.7,
        )
        response_text = response.choices[0].message.content
        return response_text
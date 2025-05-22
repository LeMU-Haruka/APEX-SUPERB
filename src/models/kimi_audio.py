import torch
import numpy as np
from src.models.base_model import BaseModel
from src.models.kimi_audio_modules.api.kimia import KimiAudio


class Kimi(BaseModel):
    def __init__(self, llm_path='moonshotai/Kimi-Audio-7B-Instruct'):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = KimiAudio(model_path=llm_path, load_detokenizer=True)

        self.sampling_params = {
            "audio_temperature": 0.8,
            "audio_top_k": 10,
            "text_temperature": 0.0,
            "text_top_k": 5,
            "audio_repetition_penalty": 1.0,
            "audio_repetition_window_size": 64,
            "text_repetition_penalty": 1.0,
            "text_repetition_window_size": 16,
            "max_new_tokens": 1024,
        }

    def chat_mode(
        self,
        audio,
        sr,
        max_new_tokens=1024,
    ):
        assert sr == 16000
        audio = audio.astype(np.float32)
        messages = [
                {"role": "user", "message_type": "audio", "content": audio}
        ]
        with torch.no_grad():
            _, response = self.model.generate(messages, **self.sampling_params, output_type="text")
        return response

    def prompt_mode(
            self,
            prompt,
            audio,
            sr,
            max_new_tokens=512,
    ):
        audio = audio.astype(np.float32)
        messages = [
            {"role": "user", "message_type": "text", "content": prompt},
            {"role": "user", "message_type": "audio", "content": audio}
        ]
        with torch.no_grad():
            _, response = self.model.generate(messages, **self.sampling_params, output_type="text")
        return response
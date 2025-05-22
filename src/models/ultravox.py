import transformers
from src.models.base_model import BaseModel


class Ultralvox(BaseModel):
    def __init__(self, llm_path='fixie-ai/ultravox-v0_5-llama-3_1-8b'):
        self.pipe = transformers.pipeline(model=llm_path, trust_remote_code=True)

    def chat_mode(
        self,
        audio,
        sr,
        max_new_tokens=1024,
    ):
        assert sr == 16000
        turns = [
            {
                "role": "system",
                "content": "You are a friendly and helpful character. You love to answer questions for people."
            },
        ]
        response = self.pipe({'audio': audio, 'turns': turns, 'sampling_rate': sr}, max_new_tokens=max_new_tokens)
        return response

    def prompt_mode(
            self,
            prompt,
            audio,
            sr,
            max_new_tokens=1024,
    ):
        assert sr == 16000
        turns = [
            {
                "role": "system",
                "content": "You are a friendly and helpful character. You love to answer questions for people. " + prompt
            },
        ]
        response = self.pipe({'audio': audio, 'turns': turns, 'sampling_rate': sr}, max_new_tokens=max_new_tokens)
        return response
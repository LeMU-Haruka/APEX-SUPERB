import uuid
from src.models.base_model import BaseModel
from src.models.step_audio2_modules.stepaudio2 import StepAudio2
from pathlib import Path
import soundfile as sf

class StepAudio2Mini(BaseModel):
    def __init__(self, llm_path='Step-Audio-2-mini'):
        self. model = StepAudio2(llm_path)

    def chat_mode(
        self,
        audio,
        sr,
        max_new_tokens=1024,
    ):
        uuid_str = str(uuid.uuid4())
        cache_audio = f'cache/stepaudio_{uuid_str}.wav'
        sf.write(cache_audio, audio, sr)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "human", "content": [{"type": "audio", "audio": cache_audio}]},
            {"role": "assistant", "content": None}
        ]
        tokens, text, _ = self.model(messages, max_new_tokens=max_new_tokens, temperature=0.1, do_sample=True)
        return text

    def prompt_mode(
            self,
            prompt,
            audio,
            sr,
            max_new_tokens=1024,
    ):
        uuid_str = str(uuid.uuid4())
        cache_audio = f'cache/stepaudio_{uuid_str}.wav'
        sf.write(cache_audio, audio, sr)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "human", "content": [{"type": "audio", "audio": cache_audio}]},
            {"role": "assistant", "content": None}
        ]
        tokens, text, _ = self.model(messages, max_new_tokens=256)
        # delete temp file
        Path(cache_audio).unlink(missing_ok=True)
        return text
    
    def process(self, item, task):
        audio = item['audio']
        sr = item['sr']
        prompt = item['instruction']
        if task == 'speech_instruct_asr' or task == 'ifeval_v1_s':
            pred = self.chat_mode(audio, sr)
        else:
            pred = self.prompt_mode(prompt, audio, sr)
        return {
            'file': item['file'],
            'prompt': prompt,
            'question': item['text'],
            'pred': pred,
            'target': item['label'],
            'kargs': item['kargs'],
        }
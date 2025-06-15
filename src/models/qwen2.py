from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from src.models.base_model import BaseModel


class Qwen2Audio(BaseModel):
    def __init__(self, llm_path='Qwen/Qwen2-Audio-7B-Instruct'):
        self.processor = AutoProcessor.from_pretrained(llm_path, cache_dir='./cache')
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(llm_path, device_map="cuda", cache_dir='./cache', torch_dtype='auto')

    def chat_mode(
        self,
        audio,
        sr,
        max_new_tokens=1024,
    ):
        assert sr == 16000
        content = [{"type": "audio", "audio_url": 'audio_url'}]
        conversation = [
            {"role": "user", "content": content},
        ]
        inputs = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios = [audio]
        inputs = self.processor(text=inputs, audios=audios, return_tensors="pt", padding=True)
        inputs = inputs.to("cuda")

        generate_ids = self.model.generate(**inputs, max_length=max_new_tokens)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response

    def prompt_mode(
            self,
            prompt,
            audio,
            sr,
            max_new_tokens=1024,
    ):
        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": ""},
                {"type": "text", "text": prompt},
            ]}
        ]
        audios = [audio]
        inputs = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        inputs = self.processor(text=inputs, audios=audios, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device)

        generate_ids = self.model.generate(**inputs, max_length=max_new_tokens)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response
    
    def process(self, item, task):
        audio = item['audio']
        sr = item['sr']
        prompt = item['instruction']
        if task == 'speech_instruct_asr':
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
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from src.models.base_model import BaseModel
import torch


class Qwen25Omni(BaseModel):
    def __init__(self, llm_path='Qwen/Qwen2.5-Omni-7B'):
        self.processor = Qwen2_5OmniProcessor.from_pretrained(llm_path)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            llm_path, 
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.model.to("cuda")

    def chat_mode(
        self,
        audio,
        sr,
        max_new_tokens=1024,
    ):
        conversation = [
            {"role": "user", 
             "content": [
                {"type": "audio", "audio": audio},
            ]}
        ]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

        inputs = self.processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        generate_ids = self.model.generate(**inputs, 
                                           max_length=max_new_tokens, 
                                           return_audio=False,
                                           eos_token_id=self.processor.tokenizer.eos_token_id,
                                           temperature=0.01
                                           )
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
            {"role": "user", 
             "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": prompt},
            ]}
        ]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

        inputs = self.processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        generate_ids = self.model.generate(**inputs, max_length=max_new_tokens, return_audio=False)
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
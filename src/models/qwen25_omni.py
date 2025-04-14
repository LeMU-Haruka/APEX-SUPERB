from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from src.models.base_model import BaseModel


class Qwen25Omni(BaseModel):
    def __init__(self, llm_path='Qwen/Qwen2-Audio-7B-Instruct'):
        self.processor = Qwen2_5OmniProcessor.from_pretrained(llm_path)
        self.model = Qwen2_5OmniModel.from_pretrained(llm_path, 
                                                      torch_dtype="auto", 
                                                      device_map="auto", 
                                                      attn_implementation="eager",
                                                      )
        self.model.to("cuda")


    def chat_mode(
        self,
        audio,
        sr,
        max_new_tokens=2048,
    ):
        content = [{"type": "audio", "audio_url": 'audio_url'}]
        conversation = [
            {"role": "user", "content": content},
        ]
        inputs = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios = [audio]
        inputs = self.processor(text=inputs, audios=audios, return_tensors="pt", padding=True)
        inputs = inputs.to("cuda")

        generate_ids = self.model.generate(**inputs, return_audio=False)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response

    def prompt_mode(
            self,
            prompt,
            audio,
            sr,
            max_new_tokens=2048,
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


    def text_mode(self, prompt, text, max_new_tokens=2048):
        content = [{"type": "text", "text": text}]
        conversation = [
            {"role": "user", "content": content},
        ]
        inputs = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        inputs = self.processor(text=inputs, audios=None, return_tensors="pt", padding=True)
        inputs = inputs.to("cuda")

        generate_ids = self.model.generate(**inputs, max_length=2048)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response


# debug


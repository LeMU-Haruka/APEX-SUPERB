from transformers import AutoModel
from src.models.base_model import BaseModel
import soundfile as sf

class DeSTA2(BaseModel):
    def __init__(self, llm_path='/userhome/models/DeSTA2-8B-beta'):
        self.model = AutoModel.from_pretrained("/userhome/models/DeSTA2-8B-beta", device_map="cuda", trust_remote_code=True)

    def chat_mode(
        self,
        audio,
        sr,
        max_new_tokens=1024,
    ):
        assert sr == 16000
        cache_audio = 'cache/desta2_temp.wav'
        sf.write(cache_audio, audio, sr)
        messages = [
                    {"role": "system", "content": "You are a helpful voice assistant."},
                    {"role": "audio", "content": cache_audio}
                ]
        generated_ids = self.model.chat(
            messages, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            temperature=0.6, 
            top_p=0.9
        )
        response = self.model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def prompt_mode(
            self,
            prompt,
            audio,
            sr,
            max_new_tokens=1024,
    ):
        cache_audio = 'cache/desta2_temp.wav'
        sf.write(cache_audio, audio, sr)
        messages = [
                    {"role": "system", "content": "You are a helpful voice assistant."},
                    {"role": "audio", "content": cache_audio},
                    {"role": "user", "content": prompt}
                ]
        generated_ids = self.model.chat(
            messages, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            temperature=0.6, 
            top_p=0.9
        )
        response = self.model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


    def text_mode(self, prompt, text, max_new_tokens=1024):
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

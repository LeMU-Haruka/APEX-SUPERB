from transformers import AutoModel
from src.models.base_model import BaseModel
import soundfile as sf
from src.models.desta25_modules import DeSTA25AudioModel


class DeSTA25(BaseModel):
    def __init__(self, llm_path='DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B'):
        self.model = DeSTA25AudioModel.from_pretrained(llm_path)
        self.model.to("cuda")


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
            {
                "role": "system",
                "content": "Focus on the audio clips and instructions."
            },
            {
                "role": "user",
                "content": "<|AUDIO|>\n",
                "audios": [{
                    "audio": cache_audio,  # Path to your audio file
                    "text": None
                }]
            }
        ]

        outputs = self.model.generate(
            messages=messages,
            do_sample=False,
            top_p=1.0,
            temperature=1.0,
            max_new_tokens=max_new_tokens
        )
        response = outputs.text
        if isinstance(response, list) and len(response) > 0:
            response = response[0]
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
        # Run inference with audio input
        messages = [
            {
                "role": "system",
                "content": "Focus on the audio clips and instructions."
            },
            {
                "role": "user",
                "content": "<|AUDIO|>\n" + prompt,
                "audios": [{
                    "audio": cache_audio,  # Path to your audio file
                    "text": None
                }]
            }
        ]

        outputs = self.model.generate(
            messages=messages,
            do_sample=False,
            top_p=1.0,
            temperature=1.0,
            max_new_tokens=max_new_tokens
        )
        response = outputs.text
        if isinstance(response, list) and len(response) > 0:
            response = response[0]
        return response


    def text_mode(self, prompt, text, max_new_tokens=1024):
        content = [{"type": "text", "text": text}]
        conversation = [
            {"role": "user", "content": content},
        ]
        inputs = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        inputs = self.processor(text=inputs, audios=None, return_tensors="pt", padding=True)
        inputs = inputs.to("cuda")

        generate_ids = self.model.generate(**inputs, max_length=max_new_tokens)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response

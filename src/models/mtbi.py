from src.models.base_model import BaseModel
from src.models.mtbi_modules.mtbi_model import MTBI
from src.models.mtbi_modules.utils import load_config


class MTBIModel(BaseModel):
    def __init__(self, llm_path='Qwen/Qwen2-Audio-7B-Instruct'):
        
        config = 'config/data_config.yaml'
        config = load_config(config)

        self.model = MTBI.load_from_checkpoint(
            llm_path,
            llama_hidden_size=config.llama_hidden_size,
            speech_model_path=config.speech_model_path,
            llama_ckpt=config.llama_path
        )


    def chat_mode(
        self,
        audio,
        sr,
        max_new_tokens=1024,
    ):
        assert sr == 16000
        response = self.model.generate(audio, max_new_tokens=max_new_tokens)
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

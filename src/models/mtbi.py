from src.models.base_model import BaseModel
from src.models.mtbi_modules.mtbi_model import MTBI
from src.models.mtbi_modules.utils import load_config


class MTBIModel(BaseModel):
    def __init__(self, llm_path='Qwen/Qwen2-Audio-7B-Instruct'):
        
        config = 'src/models/mtbi_modules/config/data_config.yaml'
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
        print(response)
        return response

    def prompt_mode(
            self,
            prompt,
            audio,
            sr,
            max_new_tokens=1024,
    ): 
        response = self.model.generate(audio, prompt=prompt, max_new_tokens=max_new_tokens)
        print(response)
        return response


    def asr(self, audio, sr):
        assert sr == 16000
        prompt = 'Transcribe the following speech to text: '
        response = self.model.generate(audio, prompt, max_new_tokens=1024)
        print(response)
        return response
    

    def build_messages(self, instruction, text):
        messages = [
            {"role": "system",
            "content": "You are a helpful assistant."},
            {"role": "user",
            "content": instruction + '' + text},
        ]
        prompt_ids = self.model.llama_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        return prompt_ids 

    def text_mode(self, instruction, text, max_new_tokens=1024):
        prompt = self.build_messages(instruction, text)
        out_ids = self.model.llama_model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.model.llama_tokenizer.eos_token_id,
        )
        response = self.model.llama_tokenizer.decode(out_ids[0][prompt.shape[-1]:], skip_special_tokens=True)
        return response


    def process(self, item, task):
        if 'asr' in task:
            response = self.asr(item['audio'], item['sr'])
        else:
            text = self.asr(item['audio'], item['sr'])
            response = self.text_mode(item['instruction'], text)

        return response
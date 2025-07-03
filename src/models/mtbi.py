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

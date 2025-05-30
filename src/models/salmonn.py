import os
import numpy as np
import torch
from src.models.base_model import BaseModel
from transformers import WhisperFeatureExtractor

from src.models.salmonn_modules.config import Config
from src.models.salmonn_modules.models.salmonn import SALMONNModel
from src.models.salmonn_modules.utils import move_to_cuda


class SALMONN(BaseModel):
    """
    Follow the guide in Hugging Face tsinghua-ee/SALMONN
    We use the vicuna 13b-v1.1 model as the LLM
    """
    def __init__(self, llm_path='tsinghua-ee/SALMONN'):
        config_path = 'src/models/salmonn_modules/configs/decode_config.yaml'
        self.cfg = Config(config_path)

        self.update_config(llm_path)       
        self.model = SALMONNModel.from_config(self.cfg.config.model)
        self.model.to('cuda')
        self.model.eval()
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(self.cfg.config.model.whisper_path)

    def update_config(self, model_path):
        self.cfg.config.model.llama_path = os.path.join(model_path, 'vicuna-13b-v1.1')
        self.cfg.config.model.beats_path = os.path.join(model_path, 'BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt') 
        self.cfg.config.model.whisper_path = os.path.join(model_path, 'whisper-large-v2')
        self.cfg.config.model.ckpt = os.path.join(model_path, 'salmonn_v1.pth')
        self.cfg.config.bert_path = os.path.join(model_path, 'bert-base-cased')

    def prepare_one_sample(self, audio, sr):
        spectrogram = self.wav_processor(audio, return_tensors="pt")["input_features"]
        if len(audio) < sr: # pad audio to at least 1s
            sil = np.zeros(sr - len(audio), dtype=float)
            audio = np.concatenate((audio, sil), axis=0)
        samples = {
            "spectrogram": spectrogram,
            "raw_wav": torch.from_numpy(audio).unsqueeze(0),
            "padding_mask": torch.zeros(len(audio), dtype=torch.bool).unsqueeze(0),
        }
        samples = move_to_cuda(samples)
        return samples

    def chat_mode(
        self,
        audio,
        sr,
        max_new_tokens=1024,
    ):
        samples = self.prepare_one_sample(audio, sr) 
        prompt = [
            self.cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> ")
        ]
        response = self.model.generate(samples, self.cfg.config.generate, prompts=prompt)[0]
        return response

    def prompt_mode(
            self,
            prompt,
            audio,
            sr,
            max_new_tokens=1024,
    ):
        samples = self.prepare_one_sample(audio, sr) 
        prompt = [
            self.cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + prompt.strip())
        ]
        response = self.model.generate(samples, self.cfg.config.generate, prompts=prompt)[0]
        return response

    def text_mode(self, prompt, text, max_new_tokens=1024):
        content = [{"type": "text", "text": text}]
        conversation = [
            {"role": "user", "content": content},
        ]
        inputs = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        inputs = self.processor(text=inputs, audios=None, return_tensors="pt", padding=True)
        inputs = inputs.to("cuda")

        generate_ids = self.model.generate(**inputs, max_length=1024)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response




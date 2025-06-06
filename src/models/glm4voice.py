import os
import torch
from transformers import AutoModel, AutoTokenizer, WhisperFeatureExtractor
from .glm_modules.speech_tokenizer.utils import extract_speech_token
from .glm_modules.speech_tokenizer.modeling_whisper import WhisperVQEncoder
from src.models.base_model import BaseModel


class Glm4Voice(BaseModel):
    def __init__(self, llm_path='THUDM/glm-4-voice-9b'):
        """
        Please download all the required model and put them in the same local folder.
        """
        self.glm_model = AutoModel.from_pretrained(
            llm_path,
            trust_remote_code=True,
            quantization_config=None,
            device_map={"": 0},
            cache_dir='./cache',
            torch_dtype=torch.bfloat16,
        ).eval()
        self.glm_tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True, cache_dir='./cache')
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(os.path.join(llm_path, "glm-4-voice-tokenizer"), cache_dir='./cache')
        self.whisper_model = WhisperVQEncoder.from_pretrained(os.path.join(llm_path, "glm-4-voice-tokenizer"), cache_dir='./cache').eval().to("cuda")


    def chat_mode(self, audio, sr, max_new_tokens=1024):
        audio_tokens = extract_speech_token(
            self.whisper_model, self.feature_extractor, [tuple([torch.from_numpy(audio).unsqueeze(0), sr])]
        )[0]
        assert len(audio_tokens) != 0
        audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
        audio_tokens = "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"
        user_input = audio_tokens
        system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. "

        inputs = f"<|system|>\n{system_prompt}"
        inputs += f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"
        inputs = self.glm_tokenizer([inputs], return_tensors="pt")
        inputs = inputs.to('cuda')

        rtn = self.glm_model.generate(**inputs, max_new_tokens=max_new_tokens)[:, inputs.input_ids.size(1):]
        text_tokens = []
        audio_offset = self.glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
        for item in rtn[0]:
            if item < audio_offset:
                text_tokens.append(item)
        return self.glm_tokenizer.decode(text_tokens, ignore_special_tokens=False)
    

    def prompt_mode(
        self,
        prompt,
        audio,
        sr,
        max_new_tokens=4096,
    ):
        audio_tokens = extract_speech_token(
            self.whisper_model, self.feature_extractor, [tuple([torch.from_numpy(audio).unsqueeze(0), sr])]
        )[0]
        assert len(audio_tokens) != 0
        audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
        audio_tokens = "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"
        user_input = audio_tokens
        system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. "


        inputs = f"<|system|>\n{system_prompt}"
        inputs += f"<|user|>\n{prompt}\n{user_input}<|assistant|>streaming_transcription\n"
        inputs = self.glm_tokenizer([inputs], return_tensors="pt")
        inputs = inputs.to('cuda')

        rtn = self.glm_model.generate(**inputs, max_new_tokens=max_new_tokens)[:, inputs.input_ids.size(1):]
        text_tokens = []
        audio_offset = self.glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
        for item in rtn[0]:
            if item < audio_offset:
                text_tokens.append(item)
        response = self.glm_tokenizer.decode(text_tokens, ignore_special_tokens=False)
        return response

    def generate_text(
        self,
        text,
    ):
        history = []
        history.append({"role": "user", "content": text})
        user_input = text
        system_prompt = "User will provide you with a text instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens."
        inputs = f"<|system|>\n{system_prompt}"
        inputs += f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"
        inputs = self.glm_tokenizer([inputs], return_tensors="pt")
        inputs = inputs.to('cuda')
        rtn = self.glm_model.generate(**inputs, max_new_tokens=4096)[:, inputs.input_ids.size(1):]
        text_tokens = []
        audio_offset = self.glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
        for item in rtn[0]:
            if item < audio_offset:
                text_tokens.append(item)
        return self.glm_tokenizer.decode(text_tokens, ignore_special_tokens=False)
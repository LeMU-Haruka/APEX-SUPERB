import torch
import torchaudio
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import ujson
import tempfile
import soundfile as sf


from src.models.base_model import BaseModel


# BAICHUAN_PATH = "baichuan-inc/Baichuan-Omni-1d5"
class BaichuanAudio(BaseModel):

    def __init__(self, llm_path='/userhome/models/Baichuan-Audio-Instruct'):
        self.role_prefix = {
            'system': '<B_SYS>',
            'user': '<C_Q>',
            'assistant': '<C_A>',
            'audiogen': '<audiotext_start_baichuan>'
        }
        self.model = AutoModelForCausalLM.from_pretrained(llm_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        self.model.training = False
        self.model.bind_processor(self.tokenizer, training=False, relative_path="/")
        self.audio_start_token = self.tokenizer.convert_ids_to_tokens(self.model.config.audio_config.audio_start_token_id)
        self.audio_end_token = self.tokenizer.convert_ids_to_tokens(self.model.config.audio_config.audio_end_token_id)
        self.special_token_partten = re.compile('<\|endoftext\|>|<audiogen_start_baichuan>|<audiogen_end_baichuan>')

    def preprocess_messages(self, messages):
        text = ""
        for i, msg in enumerate(messages):
            text += self.role_prefix[msg['role']]
            text += msg['content']
        text += self.role_prefix["assistant"]
        return text

    def chat_mode(
        self,
        audio,
        sr,
        max_new_tokens=2048,
    ):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".wav") as temp_file:
            temp_filename = temp_file.name
            # Write the audio data to the file
            sf.write(temp_file.name, audio, sr, format='wav')

        g_history = []

        g_history.append({
            "role": "system",
            "content": "You are a helpful assistant who tries to help answer the user's question."
        })

        g_history.append({
            "role": "user",
            "content": self.audio_start_token + ujson.dumps({'path': temp_filename}, ensure_ascii=False) + self.audio_end_token
        })
        message = self.preprocess_messages(g_history)
        pret = self.model.processor([message])
        plen = pret.input_ids.shape[1]
        ret = self.model.generate(
            pret.input_ids.cuda(),
            attention_mask=pret.attention_mask.cuda(),
            audios=pret.audios.cuda() if pret.audios is not None else None,
            encoder_length=pret.encoder_length.cuda() if pret.encoder_length is not None else None,
            bridge_length=pret.bridge_length.cuda() if pret.bridge_length is not None else None,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            stop_strings=['<|endoftext|>'],
            do_sample=True, temperature=0.8, top_k=20, top_p=0.85, repetition_penalty=1.1, return_dict_in_generate=True,
        )
        text_segment = self.tokenizer.decode(ret.sequences[0, plen:])
        full_text = re.sub(self.special_token_partten, '', text_segment)

        return full_text



    def prompt_mode(
        self,
        prompt,
        audio,
        sr,
        max_new_tokens=2048,
    ):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".wav") as temp_file:
            temp_filename = temp_file.name
            # Write the audio data to the file
            sf.write(temp_file.name, audio, sr, format='wav')

        g_history = []

        g_history.append({
            "role": "system",
            "content": "You are a helpful assistant. " + prompt
        })

        g_history.append({
            "role": "user",
            "content": self.audio_start_token + ujson.dumps({'path': temp_filename}, ensure_ascii=False) + self.audio_end_token
        })
        message = self.preprocess_messages(g_history)
        pret = self.model.processor([message])
        plen = pret.input_ids.shape[1]
        ret = self.model.generate(
            pret.input_ids.cuda(),
            attention_mask=pret.attention_mask.cuda(),
            audios=pret.audios.cuda() if pret.audios is not None else None,
            encoder_length=pret.encoder_length.cuda() if pret.encoder_length is not None else None,
            bridge_length=pret.bridge_length.cuda() if pret.bridge_length is not None else None,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            stop_strings=['<|endoftext|>'],
            do_sample=True, temperature=0.8, top_k=20, top_p=0.85, repetition_penalty=1.1, return_dict_in_generate=True,
        )
        text_segment = self.tokenizer.decode(ret.sequences[0, plen:])
        full_text = re.sub(self.special_token_partten, '', text_segment)

        return full_text


def load_audio(audio_path):
    wave, sr = sf.read(audio_path)
    return {'array': wave, 'sampling_rate': sr}


# wave = load_audio("/userhome/datasets/LibriSpeech/test-clean/61/70968/61-70968-0000.flac")
# model = Baichuan()
#
# output = model.generate_audio(wave)
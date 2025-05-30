from src.models.base_model import BaseModel
import argparse
import os
import torch
import math
import torchaudio.compliance.kaldi as k
from huggingface_hub import snapshot_download

from src.models.freeze_omni_modules.pipeline import inferencePipeline


class audioEncoderProcessor:
    def __init__(self, chunk_size=16):
        self.chunk_size = 16
        self.chunk_overlap = 3
        self.feat_dim = 80
        self.frame_size = 400
        self.frame_shift = 160
        self.frame_overlap = self.frame_size - self.frame_shift
        self.CHUNK = self.frame_shift * self.chunk_size
        self.reset()

    def get_chunk_size(self):
        return self.CHUNK

    def reset(self):
        self.input_chunk = torch.zeros([1, self.chunk_size + self.chunk_overlap, self.feat_dim])
        self.input_sample = torch.zeros([1, self.CHUNK + self.frame_overlap, 1])

    def fbank_shift(self, sample_data):
        # fbank feature shift
        self.input_sample[:, :self.frame_overlap, :] = self.input_sample[:, -self.frame_overlap:, :].clone()
        self.input_sample[:, self.frame_overlap:, :] = sample_data

    def chunk_data_shift(self, xs):
        # chunk feature shift
        self.input_chunk[:, :self.chunk_overlap, :] = self.input_chunk[:, -self.chunk_overlap:, :].clone()
        self.input_chunk[:, self.chunk_overlap:, :] = xs.squeeze(0)

    def process(self,
                audio: torch.Tensor):
        with torch.no_grad():
            sample_data = torch.tensor(audio).reshape(1, -1, 1)[:, :, :1] * 32768
            self.fbank_shift(sample_data)
            # use kaldi api to compute fbank
            xs = k.fbank(waveform=self.input_sample.squeeze(-1), dither=0,
                         frame_length=25, frame_shift=10, num_mel_bins=self.feat_dim)
            self.chunk_data_shift(xs)
        return self.input_chunk.clone()


class FreezeOmni(BaseModel):
    def __init__(self, llm_path='VITA-MLLM/Freeze-Omni'):
        if not os.path.exists(llm_path):
            snapshot_download(
                repo_id="VITA-MLLM/Freeze-Omni",
                local_dir=llm_path,
            )
        # can replace with your own qwen2-7B-instruct path
        # qwen2_path = os.path.join(llm_path, 'Qwen2-7B-Instruct')
        qwen2_path = '/userhome/models/Qwen2-7B-Instruct'
        if not os.path.exists(qwen2_path):
            snapshot_download(
                repo_id='Qwen/Qwen2-7B-Instruct',
                local_dir=qwen2_path,
            )

        configs = argparse.Namespace(
            model_path=os.path.join(llm_path, 'checkpoints'),
            llm_path=qwen2_path,
            top_k=20,
            top_p=0.8,
            temperature=0.8,
        )

        self.pipeline = inferencePipeline(configs)

        self.audio_processor = audioEncoderProcessor()

    def chat_mode(
        self,
        audio,
        sr,
        max_new_tokens=1024,
    ):
        wav = torch.tensor(audio)
        # Satge0: preprocess
        # set system role, stat will be set to 'sl'
        stat = 'pre'
        outputs = self.pipeline.speech_dialogue(None, stat=stat, role="You are a helpful speech assistant.")
        chunk_size = self.audio_processor.get_chunk_size()

        # Satge1: start listen
        # stat will be auto set to 'cl' after Stage1
        wav_input = torch.zeros(math.ceil(wav.shape[0] / chunk_size) * chunk_size)
        wav_input[:wav.shape[0]] = wav
        for i in range(0, wav_input.shape[0], chunk_size):
            fbank = self.audio_processor.process(wav_input[i:i + chunk_size])
            outputs = self.pipeline.speech_dialogue(fbank, **outputs)
            outputs['stat'] = 'cl'
        self.audio_processor.reset()

        outputs['adapter_cache'] = None
        outputs['encoder_cache'] = None
        outputs['pe_index'] = 0
        outputs['stat'] = 'ss'

        # Stage3: start speak
        outputs = self.pipeline.speech_dialogue(None, **outputs)

        whole_text = ''
        last_text = ''

        # Stage4: contiune speak until stat is set to 'sl'
        # use 'stop' to interrupt generation, stat need to be manually set as 'sl'
        stop = False
        while True:
            if len(outputs['past_tokens']) > max_new_tokens:
                stop = True
            if stop:
                break
            del outputs['text']
            del outputs['hidden_state']
            outputs = self.pipeline.speech_dialogue(None, **outputs)
            if outputs['stat'] == 'cs':
                whole_text += outputs['text'][len(last_text):]
                # logger.info(len(outputs['past_tokens']))
            if outputs['stat'] == 'sl':
                break
            last_text = outputs['text']

        outputs['stat'] = 'sl'
        outputs['last_id'] = None
        return whole_text


    def prompt_mode(
        self,
        prompt,
        audio,
        sr,
        max_new_tokens=1024,
    ):
        wav = torch.tensor(audio)
        # Satge0: preprocess
        # set system role, stat will be set to 'sl'
        stat = 'pre'
        instruction = "You are a helpful assistant. " + prompt
        outputs = self.pipeline.speech_dialogue(None, stat=stat, role=instruction)
        chunk_size = self.audio_processor.get_chunk_size()

        # Satge1: start listen
        # stat will be auto set to 'cl' after Stage1
        wav_input = torch.zeros(math.ceil(wav.shape[0] / chunk_size) * chunk_size)
        wav_input[:wav.shape[0]] = wav
        for i in range(0, wav_input.shape[0], chunk_size):
            fbank = self.audio_processor.process(wav_input[i:i + chunk_size])
            outputs = self.pipeline.speech_dialogue(fbank, **outputs)
            outputs['stat'] = 'cl'
        self.audio_processor.reset()

        outputs['adapter_cache'] = None
        outputs['encoder_cache'] = None
        outputs['pe_index'] = 0
        outputs['stat'] = 'ss'

        # Stage3: start speak
        outputs = self.pipeline.speech_dialogue(None, **outputs)

        whole_text = ''
        last_text = ''

        # Stage4: contiune speak until stat is set to 'sl'
        # use 'stop' to interrupt generation, stat need to be manually set as 'sl'
        stop = False
        while True:
            if len(outputs['past_tokens']) > max_new_tokens:
                stop = True
            if stop:
                break
            del outputs['text']
            del outputs['hidden_state']
            outputs = self.pipeline.speech_dialogue(None, **outputs)
            if outputs['stat'] == 'cs':
                whole_text += outputs['text'][len(last_text):]
                # logger.info(len(outputs['past_tokens']))
            if outputs['stat'] == 'sl':
                break
            last_text = outputs['text']

        outputs['stat'] = 'sl'
        outputs['last_id'] = None
        return whole_text

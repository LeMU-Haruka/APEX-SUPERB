# Use a pipeline as a high-level helper
import json
import os

from transformers import pipeline
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

WHISPER_PATH = '/userhome/models/whisper-large-v3'


class Whisper:

    def __init__(self):
        self.pipe = self.load_whisper()


    def load_whisper(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_PATH, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(WHISPER_PATH)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            return_timestamps=True, # enable long speech transcribe
        )
        return pipe

    def transcribe(self, audio):
        result = self.pipe(audio)
        return result['text']


if __name__ == '__main__':
    model = Whisper()
    path = '/userhome/datasets/mmlu'
    for root, dirs, files in os.walk(path):
        # 判断 file的后缀是不是json
        for file in files:
            if file.split('.')[-1] == 'json':
                json_file = os.path.join(path, file)
                with open(json_file) as f:
                    data = json.load(f)
                for sample in data:
                    audio_path = os.path.join(path, sample['file'].split('/')[-1])
                    text = model.transcribe(audio_path)
                    sample['whisper_v3'] = text
                    sample['file'] = audio_path.split('/')[-1]
                with open(json_file, 'w') as f:
                    json.dump(data, f)
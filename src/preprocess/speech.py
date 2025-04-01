import json
import os
import random

import soundfile as sf
import librosa
import numpy as np

from src.datasets.librispeech import LibriSpeech

# 设置种子确保可重复性
random.seed(0)
np.random.seed(0)

# 将一段语音按[0.5, 0.75, 1, 1.5, 2, 2.5]变速
# def speech_speed(wav):
#     speech_ratio = [0.5, 0.75, 1, 1.5, 2, 2.5]
#     # wav, sr = sf.read(audio_file)
#     ratio = random.choice(speech_ratio)
#     modified_wav = librosa.effects.time_stretch(wav, rate=ratio)
#     return modified_wav, ratio


if __name__ == '__main__':
    data = LibriSpeech()
    output_dir = './speech_librispeech'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_json = []
    for item in data:
        # audio = item['audio']
        # sr = item['sr']
        # modified_audio, ratio = speech_speed(audio)
        # output_path = os.path.join(output_dir, item['file'])
        # sf.write(output_path, modified_audio, sr)

        meta = {
            'sr': item['sr'],
            'file': item['file'],
            'output': item['output'],
            'file_path': item['file_path']
        }

        output_json.append(meta)
    output_json = json.dumps(output_json, indent=4)
    with open(os.path.join(output_dir, 'librispeech.json'), 'w') as f:
        f.write(output_json)

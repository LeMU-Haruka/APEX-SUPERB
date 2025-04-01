import json
import os
import pandas as pd
from torch.utils.data.dataset import Dataset

from src.config.prompts import TASK_PROMPTS
from src.datasets.utils import load_audio_file

SAMPLE_RATE = 16000
LIBRISPEECH_ROOT = '/userhome/datasets/LibriSpeech'


class LibriSpeech(Dataset):
    def __init__(self, split=['test-clean']):
        super(LibriSpeech, self).__init__()
        self.libri_root = LIBRISPEECH_ROOT

        self.files = []
        self.X_lens = []
        self.load_LibriSpeech(split)
        self.ls_trans = self.load_ls_transcript(self.files)

    def load_LibriSpeech(self, split):
        """Load Librispeech dataset"""
        meta_path = [os.path.join('./src/datasets/ls_meta', s + '.csv') for s in split]
        table_list = []
        for file in meta_path:
            if os.path.exists(file):
                table_list.append(
                    pd.read_csv(file)
                )
        table_list = pd.concat(table_list)
        file_names = table_list['file_path'].tolist()
        file_list = [os.path.join(self.libri_root, x) for x in file_names]
        self.files += file_list
        self.X_lens = table_list['length'].tolist()
        print('Total librispeech files is {}'.format(len(self.files)))

    def load_ls_transcript(self, x_list):
        """Load the transcripts for Librispeech"""

        def process_trans(transcript):
            # TODO: support character / bpe
            transcript = transcript.upper()
            return transcript

        trsp_sequences = {}
        split_spkr_chap_list = list(
            set(
                "/".join(x.split('/')[:-1]) for x in x_list
            )
        )

        for dir in split_spkr_chap_list:
            parts = dir.split('/')
            trans_path = f"{parts[-2]}-{parts[-1]}.trans.txt"
            path = os.path.join(self.libri_root, dir, trans_path)
            assert os.path.exists(path)

            with open(path, "r") as trans_f:
                for line in trans_f:
                    lst = line.strip().split()
                    trsp_sequences[lst[0]] = process_trans(" ".join(lst[1:]))

        x_names = set([self._parse_x_name(x) for x in x_list])
        y_names = set(trsp_sequences.keys())
        usage_list = list(x_names & y_names)

        trsp_sequences = {key: trsp_sequences[key] for key in usage_list}
        return trsp_sequences

    def __getitem__(self, index):
        # Load acoustic feature and pad
        file_path = self.files[index]
        text = self.ls_trans[self._parse_x_name(file_path)].lower()
        audio, sr = load_audio_file(file_path)
        file_name = file_path.split('/')[-1]

        return {
            'audio': audio,
            'output': text,
            'instruction': text,
            'prompt': TASK_PROMPTS['asr'],
            'sr': sr,
            'file': file_name,
            'file_path': file_path
        }

    def __len__(self):
        return len(self.files)

    def _parse_x_name(self, x):
        return x.split('/')[-1].split('.')[0]


data = LibriSpeech()

# preprocess
result = []
for d in data:
    result.append({
        'question': d['instruction'],
        'answer': d['output'],
        'filename': d['file'],
        'file_path': d['file_path']
    })
json_str = json.dumps(result)

with open('librispeech.json', 'w') as outfile:
    outfile.write(json_str)



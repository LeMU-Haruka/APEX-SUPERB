import json
import os
import random

import librosa
from torch.utils.data.dataset import Dataset
from datasets import load_dataset, load_from_disk

from src.datasets.task_prompts import GSM8K_FEWSHOT_RATIONALE


TASK_LIST = [
    # basic tasks
    "asr_commonvoice",
    "asr_librispeech",

    "dialogue_ser",
    "emotion_recognition",

    "animal_classification",
    "sound_classification",

    # instruction tasks
    "text_multi_instruction_st",
    "text_multi_instruction_ASR",
    "speech_multi_instruction_asr",
    "ifeval",

    # input robustness tasks
    "librispeech_noise",
    "librispeech_emotion",
    "librispeech_speed",
    "librispeech_multispeaker",

    
    # QA and reasoning tasks
    "gsm8k",
    "alpaca_empathy",
    "mmlu",
    "alpaca_eval",
    "speaker_role",
    "mmau"
]

def create_few_shot_prompt(task):
    """
    Create a few-shot prompt for the GSM8K dataset.
    """
    num_shots = int(task.split('_')[-1])
    rationales = GSM8K_FEWSHOT_RATIONALE
    prompt  = 'Here are some examples:\n'
    for i in range(num_shots):
        rationale = random.choice(rationales)
        rationale = rationale.replace('[num]', str(i + 1))
        prompt += rationale + '\n'
    prompt += 'Now, follow the same process to solve the math question in the speech:\n'
    return prompt


class HFDataset(Dataset):

    def __init__(self, task, model, local_data='./local_datasets'):
        # super(HFDataset, self).__init__(root, task)
        self.task = task
        self.model = model
        
        task = task.split('_fewshot')[0]
        data_path = os.path.join(local_data, task)
        if os.path.exists(data_path):
            self.items = load_from_disk(data_path)
        elif task in TASK_LIST:
            self.items= load_dataset(f'APEX-SUPERB/{task}', split='test')
        else:
            self.items = load_dataset(task, split='test')


    def __getitem__(self, index):
        item = self.items[index]
        # prompt = TASK_PROMPTS[self.task]
        sr = item["audio"]["sampling_rate"]
        array = item["audio"]["array"]
        # resample all audio to 16k for a fair comparison
        if sr != 16000:
            array = librosa.resample(array, orig_sr=sr, target_sr=16000)
        if 'text' not in item:
            item['text'] = item['instruction']

        if 'fewshot' in self.task:
            item['instruction'] = create_few_shot_prompt(self.task)
        item['sr'] = 16000
        item['audio'] = array
        item['kargs'] = {}
        for key in item.keys():
            if key not in ['audio', 'instruction', 'file', 'text', 'label', 'kargs', 'sr']:
                item['kargs'][key] = item[key]
        return item

    def __len__(self):
        return len(self.items)
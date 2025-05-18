import json
import os
import random

import librosa
from torch.utils.data.dataset import Dataset
from datasets import load_dataset, load_from_disk

from src.datasets.task_prompts import GSM8K_FEWSHOT_RATIONALE
from src.datasets.utils import load_audio_file



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




# class BaseDataset(Dataset):
#     def __init__(self, args):
#         super(BaseDataset, self).__init__()
#         from src.datasets import DATASET_PATH_CONFIG

#         self.root = DATASET_PATH_CONFIG[args.task]
#         self.task = args.task
#         self.model = args.model
#         file_list = []
#         if not os.path.isdir(self.root):
#             file_list.append(self.root)
#             # 去除最后的文件名，保留前面的路径
#             self.root = os.path.dirname(self.root)
#         else:
#             for index, dir, files in os.walk(self.root):
#                 for file in files:
#                     if file.endswith('.json'):
#                         json_path = os.path.join(self.root, file)
#                         file_list.append(json_path)
#         print('Find {} meta files'.format(len(file_list)))
#         self.items = self.load_data(file_list)


#     def load_data(self, file_list):
#         total_data = []
#         for file in file_list:
#             data = json.load(open(file))
#             total_data += data
#         return total_data

#     def __getitem__(self, index):
#         item = self.items[index]
#         audio, sr = load_audio_file(os.path.join(self.root, item['filename']))
#         if sr != 16000:
#             audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
#         item['sr'] = 16000
#         item['audio'] = audio
#         # 判断item是否包含prompt字段
#         if 'prompt' not in item:
#             prompt = TASK_PROMPTS[self.task]
#             item['prompt'] = prompt
#         item['kargs'] = {}
#         for key in item.keys():
#             if key not in ['audio', 'prompt', 'filename', 'question', 'answer', 'kargs']:
#                 item['kargs'][key] = item[key]
#         return item

#     def __len__(self):
#         return len(self.items)


class HFDataset(Dataset):

    def __init__(self, task, model, local_data='./local_datasets'):
        # super(HFDataset, self).__init__(root, task)
        self.task = task
        self.model = model
        
        task = task.split('_fewshot')[0]
        data_path = os.path.join(local_data, task)
        if os.path.exists(data_path):
            # 如果本地数据集存在，则使用本地数据集
            self.items = load_from_disk(data_path)
        else:
            # 否则使用huggingface上的数据集
            self.items= load_dataset(f'APEX-SUPERB/{task}', split='test')

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

# class HFChoiceDataset(BaseDataset):

#     def __init__(self, root, task):
#         # super(HFDataset, self).__init__(root, task)
#         self.task = task
#         self.items = load_dataset(root, split='test')

#     def __getitem__(self, index):
#         item = self.items[index]
#         # prompt = TASK_PROMPTS[self.task]
#         item['audio'] = item['audio']['array']
#         item['prompt'] = item['instruction']
#         choice = item['choices']
#         # 将 choice的每一个选项转换成 A.xxx B.xxx C.xxx D.xxx 的字符串形式


#         item['prompt'] = item['prompt'] + ' ' + choice
#         item['filename'] = item['id']
#         if 'transcript' in item:
#             item['question'] = item['transcript']
#         else:
#             item['question'] = 'no transcript'
#         return item


# class ChoiceDataset(BaseDataset):

#      def __getitem__(self, index):
#          item = self.items[index]
#          prompt = TASK_PROMPTS[self.task]
#          choice = item['choice']
#          item['prompt'] = prompt + choice
#          audio, sr = load_audio_file(os.path.join(self.root, item['filename'].split('/')[-1]))
#          item['audio'] = audio
#          return item



# 测试所有数据集的读取情况

# root = '/userhome/datasets/SGBench/librispeech_multispeaker'
# dataset = BaseDataset(root, 'asr')
# # dataset = ChoiceDataset(root, 'mmlu')
# print(len(dataset))
# for item in dataset:
#     print(item)
#     break

# import pandas as pd

# meta = '/userhome/datasets/SGBench/common_voice17_test/test.tsv'
# data = pd.read_csv(meta, sep='\t')

# json_data = []

# for index, row in data.iterrows():
#     json_data.append({
#         'filename': row['path'],
#         'question': row['sentence'],
#         'output': row['sentence'],

#     })

# json_str = json.dumps(json_data, indent=4)
# with open('/userhome/datasets/SGBench/common_voice17_test/test.json', 'w') as f:
#     f.write(json_str)


# root = '/userhome/datasets/SGBench/common_voice17_test/en_test_0'
# json_data = json.load(open(os.path.join(root, 'common_voice17.json')))
# for index, item in enumerate(json_data):
#     utt = item['output']
#     item['answer'] = utt
#     item.pop('output')

# json_str = json.dumps(json_data, indent=4)
# with open(os.path.join(root, 'common_voice17.json'), 'w') as f:
#     f.write(json_str)
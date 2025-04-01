import json
import os

from torch.utils.data.dataset import Dataset
from datasets import load_dataset

from src.config.prompts import TASK_PROMPTS
from src.datasets.utils import load_audio_file


class TaskDataset(Dataset):
    def __init__(self, root, task):
        super(TaskDataset, self).__init__()
        self.root = root
        self.task = task
        file_list = []
        for index, dir, files in os.walk(root):
            for file in files:
                if file.endswith('.json'):
                    json_path = os.path.join(root, file)
                    file_list.append(json_path)
        print('Find {} meta files'.format(len(file_list)))
        self.items = self.load_data(file_list)


    def load_data(self, file_list):
        total_data = []
        for file in file_list:
            data = json.load(open(file))
            total_data += data
        return total_data

    def __getitem__(self, index):
        prompt = TASK_PROMPTS[self.task]
        item = self.items[index]
        audio, sr = load_audio_file(os.path.join(self.root, item['file'].split('/')[-1]))
        item['audio'] = audio
        item['prompt'] = prompt
        return item

    def __len__(self):
        return len(self.items)


class HFDataset(TaskDataset):

    def __init__(self, root, task):
        super(HFDataset, self).__init__(root, task)
        self.task = task
        self.items = load_dataset(root)

    def __getitem__(self, index):
        item = self.items[index]
        prompt = TASK_PROMPTS[self.task]
        item['prompt'] = prompt
        item['output'] = item['abstract']
        return item




class ChoiceDataset(TaskDataset):

     def __getitem__(self, index):
         item = self.items[index]
         prompt = TASK_PROMPTS[self.task]
         choice = item['choice']
         item['prompt'] = prompt + choice
         item['output'] = item['answer']
         audio, sr = load_audio_file(os.path.join(self.root, item['file'].split('/')[-1]))
         item['audio'] = audio
         return item


import os
from datasets import load_dataset
from .base_dataset import TASK_LIST

output_dir = 'local_datasets'
os.makedirs(output_dir, exist_ok=True)

for task in TASK_LIST:
    if os.path.exists(os.path.join(output_dir, task)):
        print(f'{task} already exists, skip download')
        continue
    repo = f'LeMUHaruka/{task}'
    dataset = load_dataset(repo, split='test')

    dataset.save_to_disk(os.path.join(output_dir, task))
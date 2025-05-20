import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from datasets import load_dataset
from src.datasets.base_dataset import TASK_LIST

output_dir = 'local_datasets'
os.makedirs(output_dir, exist_ok=True)

for task in TASK_LIST:
    if os.path.exists(os.path.join(output_dir, task)):
        print(f'{task} already exists, skip download')
        continue
    repo = f'LeMUHaruka/{task}'
    dataset = load_dataset(repo, split='test')

    dataset.save_to_disk(os.path.join(output_dir, task))
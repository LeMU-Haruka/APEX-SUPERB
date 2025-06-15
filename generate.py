import os
from argparse import ArgumentParser

# from src.datasets import datasets_map
from src.datasets.base_dataset import HFDataset
from src.models import models_map
import json
from tqdm import tqdm

from utils import set_seed

# for some devices may need to set this
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_flash_sdp(False)

set_seed(42)


def main():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen2-audio', choices=list(models_map.keys()))
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--task', type=str, default='asr')
    args = parser.parse_args()

    output_dir = args.model + '_result'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    task = args.task

    # print parameters
    for key, value in vars(args).items():
        print(f'{key}: {value}')

    # load dataset
    data = HFDataset(task, args.model)

    # load model
    model = models_map[args.model](args.model_path)


    results = []
    for item in tqdm(data, total=len(data)):
        result = model.process(item, task)
        results.append(result)

    task = results[0]['kargs']['task']
    # save results
    output_file = f'{args.model}-{task}.json'
    json_str = json.dumps(results, indent=4)  
    with open(os.path.join(output_dir, output_file), 'w') as f:
        f.write(json_str)
    print('Evaluate finished! model: {}, task: {}, output_rsult: {}'.format(args.model, task, os.path.join(output_dir, output_file)))

if __name__ == '__main__':
    main()

import os
from argparse import ArgumentParser
import torch

# from src.datasets import datasets_map
from src.datasets.base_dataset import HFDataset
from src.models import models_map
import json
from tqdm import tqdm

from src.utils import set_seed

# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_flash_sdp(False)

set_seed(42)

# def set_gpu(gpu_id):
#     """
#     设置要使用的 GPU。

#     Args:
#         gpu_id: 要使用的 GPU 的 ID (整数)。
#     """
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
#     print(f"Set CUDA_VISIBLE_DEVICES to: {os.environ['CUDA_VISIBLE_DEVICES']}")


def main():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen2-audio', choices=list(models_map.keys()))
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--task', type=str, default='asr')
    # parser.add_argument('--output_dir', type=str, default='./result')
    # parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    output_dir = args.model + '_result'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    task = args.task
    # 遍历打印参数
    for key, value in vars(args).items():
        print(f'{key}: {value}')

    # if task == 'asr':
    #     data = load_asr_data()
    # else:
    data = HFDataset(task, args.model)
    # load data
    # data = load_dataset('hlt-lab/voicebench', args.data, split=args.split)
    # data = data.cast_column("audio", Audio(sampling_rate=16_000))

    # load model
    model = models_map[args.model](args.model_path)
    # data = data.select([0,1,2,3,4,5])

    # if args.modality == 'ttft':
    #     # avoid cold start
    #     _ = model.generate_ttft(data[0]['audio'])

    # inference
    results = []
    for item in tqdm(data, total=len(data)):
        audio = item['audio']
        prompt = item['instruction']
        sr = item['sr']
        pred = model.prompt_mode(prompt, audio, sr)
        results.append({
            'file': item['file'],
            'prompt': prompt,
            'question': item['text'],
            'pred': pred,
            'target': item['label'],
            'kargs': item['kargs'],
        })

    # save results
    output_file = f'{args.model}-{args.task}.json'
    json_str = json.dumps(results, indent=4)  # Convert list of dictionaries to JSON string
    with open(os.path.join(output_dir, output_file), 'w') as f:
        f.write(json_str)
    print('Evaluate finished! model: {}, task: {}, output_rsult: {}'.format(args.model, task, os.path.join(output_dir, output_file)))

if __name__ == '__main__':
    main()

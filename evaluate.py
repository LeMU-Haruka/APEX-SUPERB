from argparse import ArgumentParser
import json
import os

from src.evaluation import EVALUATOR_MAP


def main():
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='qwen2-audio')
    parser.add_argument('--evaluator', type=str, required=True,  choices=list(EVALUATOR_MAP.keys())),
    parser.add_argument('--output_dir', type=str, default='./evaluate_result')
    parser.add_argument('--api', type=str, default='vllm', choices=['no', 'gemini', 'gpt', 'vllm'])
    parser.add_argument('--align', action='store_true', help='Whether to align the text with LLM')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('./cache', exist_ok=True)

    evaluator_type = args.evaluator
    meta_file = args.file
    task_name = meta_file.split('/')[-1].split('.')[0]
    with open(meta_file, 'r') as f:
        data = json.load(f)
    # evaluator = EVALUATOR_MAP[args.task]()
    evaluator = EVALUATOR_MAP[evaluator_type](args.model_name, meta_file, evaluator_type, args.api, is_align=args.align)
    result = evaluator.evaluate(data)
    print(result)

    # Save the result to a file
    json_result = json.dumps(result, indent=4)
    with open(os.path.join(args.output_dir, f'{task_name}_result.json'), 'w') as f:
        f.write(json_result)



if __name__ == "__main__":
    main()

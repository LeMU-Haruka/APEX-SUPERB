from argparse import ArgumentParser
import json
import os
import time

from src.evaluation import load_evaluator
from utils import load_result_files


def main():
    parser = ArgumentParser()
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='qwen2-audio')
    parser.add_argument('--evaluator', type=str,  default=None),
    parser.add_argument('--output_dir', type=str, default='./evaluate_result')
    parser.add_argument('--api', type=str, default='vllm', choices=['gemini', 'gpt', 'vllm'])
    parser.add_argument('--align', action='store_true', help='Whether to align the output to label using LLM')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('./cache', exist_ok=True)

    result_file = args.result_path
    model_name = args.model_name


    if os.path.isfile(result_file):
        evaluated_files = [result_file]
        evaluator_type = args.evaluator
    else:
        evaluated_files = load_result_files(result_file)
        evaluator_type = args.evaluator
        if len(evaluated_files) == 0:
            raise ValueError(f"No files found in {result_file}")
        
    
    print (f"Evaluation start")
    start_time = time.time()
    
    results = []
    for file in evaluated_files:
        with open(file, 'r') as f:
            data = json.load(f)
        if evaluator_type is None:
            task = file.split('-')[-1].split('.')[0]
        else:
            task = evaluator_type
        evaluator = load_evaluator(task, model_name, file, args.api, is_align=args.align)
        result = evaluator.evaluate(data)
        results.append(result)
        print(f"Evaluated {file} end, time cost: {time.time() - start_time:.2f}s")
        print(result)

        # Save the result to a file after each task, avoid losing all results if the process is interrupted
        json_result = json.dumps(results, indent=4)
        with open(os.path.join(args.output_dir, f'{model_name}_evaluated.json'), 'w') as f:
            f.write(json_result)
    print(f"Evaluation end, total time cost: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()

from argparse import ArgumentParser
import json
from src.evaluation import task_map


def main():
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--task', type=str, required=True, choices=list(task_map.keys()))
    args = parser.parse_args()
    data = []
    with open(args.file, 'r') as f:
        for line in f:
            json_obj = json.loads(line.strip())  # Convert JSON string to dictionary
            data.append(json_obj)
    evaluator = task_map[args.task]()
    print(evaluator.evaluate(data))



if __name__ == "__main__":
    main()


import json

from tqdm import tqdm

from utils import extract_json


ACCURACY_PROMPT = """
    You are an expert evaluator for large language models.
    Your task is to evaluate the factual accuracy of the response and its consistency with the ground truth.
    
    Instruction: [INSTRUCTION]
    
    Input text: [QUESTION]
    
    Response: [PRED]
    
    Ground Truth: [LABEL]
    
    Evaluation Criteria:
    - Is Same: Does the response convey the same meaning/result/value as the ground truth? (1 for yes, 0 for no)

    YOU MUST ONLY ***JSON FORMAT*** EVALUATION:
    {
        "is_same": <0 or 1>
    }
"""


def accuracy_metric(predictions, references):
    """
    Calculate the accuracy of predictions against references.
    :param predictions: List of predicted values
    :param references: List of true values
    :return: Accuracy as a float
    """
    correct = sum(p == r for p, r in zip(predictions, references))
    # 返回dict格式
    return {
        'accuracy': correct / len(predictions) if predictions else 0.0,
        'correct': correct,
        'total': len(predictions)
    }


def accuracy_metric_with_llm(client, data):
    """
    Calculate the accuracy of predictions against references.
    :param is_same_list: List of predicted values
    :return: Accuracy as a float
    """
    correct = 0
    failed = 0
    for item in tqdm(data, total=len(data), desc="Accuracy with LLM"):
        prompt = build_prompt(item)
        response = client.generate_response(prompt)
        json_str = extract_json(response)
        item['metric_response'] = response
        try:
            is_same = json.loads(json_str)['is_same']
            item['is_same'] = is_same
        except BaseException as e:
            print("Response formant error")
            print(e)
            print(response)
            print('#' * 20)
            print(json_str)
            item['is_same'] = -1
            failed += 1
            continue
        correct += is_same
    print(f'Total failed item is {failed}')
    return correct / (len(data) - failed)

def build_prompt(item):
    prompt = ACCURACY_PROMPT
    prompt = prompt.replace("[INSTRUCTION]", item['prompt'])
    prompt = prompt.replace("[QUESTION]", item['question'])
    prompt = prompt.replace("[PRED]", item['pred'])
    prompt = prompt.replace("[LABEL]", item['target'])
    return prompt

def build_message(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    return messages
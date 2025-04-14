import json

from utils import extract_json


IFEVAL_PROMPT = """
    You are an evaluator for instruction-following tasks in LLMs.
    Given a question, an instruction, and a response, evaluate according to the following rules:

    Instruction Following:

    Output 1 if the response fully follows the instruction, otherwise output 0.

    Instructions may involve various constraints (e.g., number of words, bullet points, JSON format, special casing, etc.).

    If the instruction requires a chain-of-thought (CoT) reasoning (e.g., "think step-by-step"), treat the presence of a clear multi-step reasoning process as mandatory for compliance.

    If CoT is missing when required, mark instruction_following = 0.

    Response Score:

    Score the overall quality of the response from 1 (very poor) to 5 (excellent), based on:

        Correctness

        Completeness

        Relevance to the question

        Clarity

    Reasons:

    For each judgment above, provide a short, precise reason.

    Output strictly in the following JSON format:
    {
        "instruction_following": 0 or 1,
        "reason_compliance": <reason>,
        "response_score": 1 to 5,
        "reason_score": <reason>
    }
    Here is the inputs:
    Instruction: [INSTRUCTION]
    Question: [QUESTION]
    Response: [RESPONSE]
"""


def build_ifeval_prompt(item):
    """
    构建 GPT 提示词
    Args:
        item: 数据项
    Returns:
        str: 提示词
    """
    instruction = item['prompt']
    question = item['question']
    response = item['pred']
    prompt = IFEVAL_PROMPT.replace('[INSTRUCTION]', instruction).replace('[QUESTION]', question).replace('[RESPONSE]', response)
    return prompt


def ifeval_metric(client, data):
    if_rate = 0
    content_score = 0

    for item in data:
        prompt = build_ifeval_prompt(item)
        response = client.generate_response(prompt)
        json_str = extract_json(response)
        try:
            json_response = json.loads(json_str)
            instruction_score = json_response['instruction_compliance']
            relevance = json_response['Response_Score']
        except BaseException as e:
            print("Response formant error")
            print(e)
            print(response)
            print('#' * 20)
            print(json_str)
            continue
        if_rate += instruction_score
        content_score += relevance
    if_rate = if_rate / len(data)
    content_score = content_score / len(data)
    fluency_score = fluency_score / len(data)
    return {
        'scores': {
            'if_rate': if_rate,
            'content_score': content_score,
        },
    }



import json

from tqdm import tqdm
import nltk
from src.evaluation.metrics.gpt_eval import build_content_score_prompt
from utils import extract_json

CACHE_DIR = 'cache'
IFEVAL_PROMPT = """
    You are an evaluator for instruction-following tasks in LLMs.
    Given a question, an instruction, and a response, evaluate according to the following rules:

    Instruction Following:

    Measure how well the response follows the instruction. The instruction may include specific requirements such as:
    - Number of words
    - JSON format
    - Uppercase or lowercase
    - Chain-of-thought (CoT) reasoning
    
    Output 1 if the response fully follows the requirement (e.g., number of words, bullet points, JSON format, special casing, etc.). of instruction, otherwise output 0.

    If the instruction requires a chain-of-thought (CoT) reasoning (e.g., "think step-by-step"), treat the presence of a clear multi-step reasoning process as mandatory for compliance.

    If CoT is missing when required, mark instruction_following = 0.

    For the instruction following score should not care about the correctness of the response. ONLY care about whether the response follows the instruction.

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

CHECK_COT_PROMPT = """
    You are an evaluation expert. Your task is to decide whether the response of LLM explicitly contains a Chain‑of‑Thought (CoT) reasoning process.

    # CRITERIA
    A text **contains** CoT if it shows any of these signs:
    1. Explicit step‑by‑step reasoning (e.g. “Step 1 … Step 2 …”).
    2. A phrase like “Let’s think step by step” followed by multi‑line reasoning.
    3. Intermediate assumptions, calculations, or conclusions before the final answer.
    4. Visible reasoning traces such as listed assumptions, conditions, or formulas.

    If the text only gives a direct answer or mere facts with no reasoning steps, it **does not contain** CoT.

    # OUTPUT FORMAT
    Respond with **exactly one** JSON object—nothing else:

    - If CoT is present: `{"contain": 1}`
    - If CoT is absent:  `{"contain": 0}`

    # INPUT RESPONSE
    [REPONSE]
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


def check_capitalization(text):
    """
    检查文本是否全部为大写字母
    Args:
        text: 文本
    Returns:
        bool: 是否包含大写字母
    """
    if text.isupper():
       return 1
    return 0

def check_length(text):
  """Counts the number of words."""
  tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
  tokens = tokenizer.tokenize(text)
  num_words = len(tokens)
  if num_words > 100:
     return 0
  return 1


def check_json(text):
    post_test = extract_json(text)
    try:
      json.loads(post_test)
    except BaseException as _:
      return 0
    return 1


def get_content_score(client, item):
    prompt = build_content_score_prompt(item)
    response = client.generate_response(prompt)
    json_str = extract_json(response)
    try:
        json_response = json.loads(json_str)
        score = json_response['overall_score']
    except BaseException as e:
        print("Response formant error")
        print(e)
        print(response)
        print('#' * 20)
        print(json_str)
        score = -1
    return score

def check_cot(client, text):
    prompt = CHECK_COT_PROMPT
    prompt = prompt.replace('[REPONSE]', text)
    response = client.generate_response(prompt)
    json_str = extract_json(response)
    return json_str

def ifeval_metric(client, data):
    if_rate = 0
    content_score = 0

    for item in tqdm(data, total=len(data), desc="IFEVAL"):
        prompt = build_ifeval_prompt(item)
        response = client.generate_response(prompt)
        json_str = extract_json(response)
        item['metric_response'] = response
        try:
            json_response = json.loads(json_str)
            instruction_score = json_response['instruction_following']
            response_score = json_response['response_score']
        except BaseException as e:
            print("Response formant error")
            print(e)
            print(response)
            print('#' * 20)
            print(json_str)
            item['if_rate'] = -1
            item['content_score'] = -1
            continue
        if_rate += instruction_score
        content_score += response_score
        item['if_rate'] = instruction_score
        item['content_score'] = response_score
    if_rate = if_rate / len(data)
    content_score = content_score / len(data)
    return {
        'scores': {
            'if_rate': if_rate,
            'content_score': content_score,
        },
    }

def check_choice(text, choice=['student', 'teacher', 'doctor', 'police', 'engineer']):
    for c in choice:
        if c in text:
            return 1
    return 0

def ifeval_metric_v1(client, data):
    total_if_rate = 0
    total_content_score = 0
    if_rate_failed = 0
    content_failed = 0
    for item in tqdm(data, total=len(data), desc="IFEVAL"):
        category = item['kargs']['category']
        pred = item['pred']
        if category == 'All capital':
            if_rate = check_capitalization(pred)
        if category == 'JSON format':
            if_rate = check_json(pred)
        if category == 'Length Constraint':
            if_rate = check_length(pred)
        if category == 'Choice':
            if_rate = check_choice(pred)
        if category == 'CoT':
            response = check_cot(client, pred)
            try:
                result = json.loads(response)
                if_rate = result['contain']
            except BaseException as e:
                print("Response formant error")
                print(e)
                print(response)
                result = 0
                if_rate_failed += 1
                item['if_rate'] = -1
        contain_score = get_content_score(client, item)
        if contain_score == -1:
            item['content_score'] = -1
            content_failed += 1
            contain_score = 0
        else:
            item['content_score'] = contain_score
        item['if_rate'] = if_rate
        total_content_score += contain_score
        total_if_rate += if_rate
    
    avg_score = total_content_score / (len(data) - content_failed)
    avg_if_rate = total_if_rate / (len(data) - if_rate_failed)
    return {
        'scores': {
            'if_rate': avg_if_rate,
            'content_score': avg_score,
        },
    }
    

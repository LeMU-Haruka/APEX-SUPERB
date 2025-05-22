
import json

from tqdm import tqdm

from utils import extract_json

CACHE_DIR = 'cache'


GPT_CONTENT_SCORE_PROMPT = """
    You are an expert evaluator for natural language generation tasks.

    Given the following **question** and the **model's prediction (pred)**, rate the prediction on three aspects: **fluency**, **relevance**, and **overall_score**, each on a scale of 1 to 5. Provide a short justification for each score.

    Scoring criteria:
    - **Fluency**: Is the language grammatically correct, natural, and easy to understand?
    - **Relevance**: Does the response answer the question correctly and appropriately?
    - **Overall_score**: How well does the response fulfill the user's intent overall?

    Return your response in this JSON format:
    {
    "fluency": [1-5],
    "relevance": [1-5],
    "overall_score": [1-5],
    "details": "[simple reason for overall score in ONLY ONE sentence]"
    }

    Here is an example:
    question: What is some cool music from the 1920s?
    pred: I'm sorry, but I cannot engage in discussions regarding political matters. My primary function is to provide assistance on a wide range of non-political subjects. If you have any other inquiries, please feel free to ask.

    Your response should like this:
    {
        "fluency": 4,
        "relevance": 1,
        "overall_score": 1,
        "details": "Even though the language is fine, it completely fails to answer the question."
    }

    Here is the inputs:
    Question: [QUESTION]
    Model response: [PRED]
    Your response: 
"""

GPT_EMPATHY_SCORE_PROMPT = """
    You will be provided with a response generated based on the following input text and emotional information:
    Input Text: [QUESTION]
    Emotional Information: [EMOTION]

    Your task is to assess the quality of the response in relation to the emotional context and intent of the original question. Please evaluate the response based on the following criteria and provide a score from 1 (poor) to 5 (excellent) for each:
    1. Empathy: How effectively does the response acknowledge and respond to the emotional state of the questioner?
    2. Content: How well does the response stay on-topic and provide an appropriate reply to the question?
    3. Clarity: How clear and easy to understand is the response?


    Generated Response: [PRED]
    Your *entire response* MUST be a single, valid JSON string.

    Use the following JSON string precisely:
    {
        "score": {
            "empathy": <score>,
            "content": <score>,
            "clarity": <score>
        },
        "details": {
            "empathy": <simple reason>,
            "content": <simple reason>,
            "clarity": <simple reason>
        }
    }
"""


def gpt_content_score(client, data):
    """
    使用 GPT 计算分数
    Args:
        client: GPT 客户端实例
        prompt: 提示词
    Returns:
        float: GPT 计算的分数
    """
    fluency_score = 0
    relevance_score = 0
    overall_score = 0
    failed = 0
    for item in tqdm(data, total=len(data), desc="GPT Content Score"):
        prompt = build_content_score_prompt(item)
        response = client.generate_response(prompt)
        json_str = extract_json(response)
        item['metric_response'] = response
        try:
            json_response = json.loads(json_str)
            fluency = json_response['fluency']
            relevance = json_response['relevance']
            overall = json_response['overall_score']
        except BaseException as e:
            print("Response formant error")
            print(e)
            print(response)
            print('#' * 20)
            print(json_str)
            failed += 1
            item['fluency'] = -1
            item['relevance'] = -1
            item['overall'] = -1
            continue
        
        fluency_score += fluency
        relevance_score += relevance
        overall_score += overall
        item['fluency'] = fluency
        item['relevance'] = relevance
        item['overall'] = overall


    print(f'Total failed item is {failed}')
    fluency_score = fluency_score / (len(data) - failed) 
    relevance_score = relevance_score / (len(data) - failed)
    overall_score = overall_score / (len(data) - failed)
    return {
        'scores': {
            'fluency': fluency_score,
            'relevance': relevance_score,
            'overall': overall_score
        },
    }

def gpt_empathy_score(client, data):
    """
    使用 GPT 计算分数
    Args:
        client: GPT 客户端实例
        prompt: 提示词
    Returns:
        float: GPT 计算的分数
    """
    empathy_score = 0
    content_score = 0
    clarity_score = 0
    failed = 0
    for item in tqdm(data, total=len(data), desc="GPT Empathy Score"):
        prompt = build_empathy_prompt(item)
        response = client.generate_response(prompt)
        json_str = extract_json(response)
        item['metric_response'] = response
        try:
            json_response = json.loads(json_str)
            empathy = json_response['score']['empathy']
            content = json_response['score']['content']
            clarity = json_response['score']['clarity']
        except BaseException as e:
            print("Response formant error")
            print(e)
            print(response)
            failed += 1
            item['empathy'] = -1
            item['content'] = -1
            item['clarity'] = -1
            continue
        empathy_score += empathy
        content_score += content
        clarity_score += clarity

    print(f'Total failed item is {failed}')
    empathy_score = empathy_score / (len(data) - failed) 
    content_score = content_score / (len(data) - failed) 
    clarity_score = clarity_score / (len(data) - failed) 
    scores = {
        'empathy': empathy_score,
        'content': content_score,
        'clarity': clarity_score,
        'overall': (empathy_score + content_score + clarity_score) / 3
    }
    return {
        'scores': scores,
    }

def build_empathy_prompt(item):
    prompt = GPT_EMPATHY_SCORE_PROMPT
    prompt = prompt.replace("[QUESTION]", item['question'])
    prompt = prompt.replace("[Emotion]", 'happy')
    prompt = prompt.replace("[PRED]", item['pred'])
    return prompt

def build_content_score_prompt(item):
    prompt = GPT_CONTENT_SCORE_PROMPT
    prompt = prompt.replace("[PROMPT]", item['prompt'])
    prompt = prompt.replace("[QUESTION]", item['question'])
    prompt = prompt.replace("[PRED]", item['pred'])
    return prompt
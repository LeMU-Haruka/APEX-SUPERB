import re
import jiwer
from tqdm import tqdm



asr_align_prompt = """
You are a text processing assistant. Your task is to clean the provided text by removing any extraneous, redundant, or non-essential expressions while preserving the core semantic content. This includes eliminating introductory statements, irrelevant formatting elements, unnecessary punctuation, or any additional commentary that does not affect the meaning.

For example, if given the input:
-------------------------------------------------
"The original content of this audio is: 'Yesterday you were trembling for a health that is dear to you, today you fear for your own, tomorrow it will be anxiety about money, the day after tomorrow the diatribe of a slanderer, the day after that the misfortune of some friend, then the prevailing weather, then something that has been broken or lost, then a pleasure with which your conscience and your vertebral column rebel.'"
-------------------------------------------------
The expected cleaned output should be:
-------------------------------------------------
"yesterday you were trembling for a health that is dear to you to day you fear for your own to morrow it will be anxiety about money the day after to morrow the diatribe of a slanderer the day after that the misfortune of some friend then the prevailing weather then something that has been broken or lost then a pleasure with which your conscience and your vertebral column reproach you again the course of public affairs"
-------------------------------------------------

This prompt should be applicable in all cases—whether the task involves translation, processing multiple-choice options, or any similar scenario where extra expressions are present. Only output the cleaned text.

Now, please process the following text:
-------------------------------------------------
[INPUT]
-------------------------------------------------
Output only the cleaned text.
"""


asr_eval_prompt = """
    You are an expert evaluator for large language models. Your task is to evaluate the quality of the response based on the following criteria:
    Input text: [text]
    Response: [target]
    Evaluation Criteria:
    Word Error Rate (WER): The percentage of words in the response that are incorrect compared to the ground truth.
    Character Error Rate (CER): The percentage of characters in the response that are incorrect compared to the ground truth.
    Provide your evaluation in the following JSON format:
    {
        "wer": <score>,
        "cer": <score>
    }
"""


def wer_metric(preds, targets):
    # 计算平均wer和cer
    WER = 0.
    CER = 0.
    for p, t in tqdm(zip(preds, targets), total=len(preds), desc="Calculating WER and CER"):
        wer = calculate_wer(p, t)
        cer = calculate_cer(p, t)
        WER += wer
        CER += cer
    WER /= len(preds)
    CER /= len(preds)
    print(f"Average WER: {WER:.3f}, Average CER: {CER:.3f}")
    # 返回dict格式
    return {
        'wer': WER,
        'cer': CER
    }

def normalize_text(text):
    """小写 + 去标点 + 多空格合并"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # 去除标点符号
    text = re.sub(r'\s+', ' ', text).strip()  # 合并多余空格
    return text

def calculate_wer(predicted_sentence, ground_truth):
    """
    计算单词错误率 (Word Error Rate - WER).

    Args:
        predicted_sentence (str):  ASR 模型预测的句子.
        ground_truth (str):  真实的句子 (ground truth).

    Returns:
        float:  WER 值 (0.0 表示完全正确, 1.0 表示完全错误).
    """

    transform = jiwer.Compose([
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.ToLowerCase()
    ])
    gt = transform(ground_truth)
    pred = transform(predicted_sentence)
    wer = jiwer.wer(gt, pred)
    return wer


def calculate_cer(predicted_sentence, ground_truth):
    """
    计算字符错误率 (Character Error Rate - CER).

    Args:
        predicted_sentence (str):  ASR 模型预测的句子.
        ground_truth (str):  真实的句子 (ground truth).

    Returns:
        float:  CER 值 (0.0 表示完全正确, 1.0 表示完全错误).
    """
    transform = jiwer.Compose([
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.ToLowerCase()
    ])
    gt = transform(ground_truth)
    pred = transform(predicted_sentence)
    cer = jiwer.cer(gt, pred)
    return cer
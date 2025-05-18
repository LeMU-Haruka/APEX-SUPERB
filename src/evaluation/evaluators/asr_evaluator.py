

import json
import os
import time

from tqdm import tqdm
from src.evaluation.api import CLIENT_MAP
from src.evaluation.evaluators.evaluator import Evaluator
from src.evaluation.metrics.wer import wer_metric


# ASR_ALIGNMENT_PROMPT = """
#         You are a text processing assistant. Your task is to clean the provided text by removing any extraneous, redundant, or non-essential expressions while preserving the core semantic content. This includes eliminating introductory statements, irrelevant formatting elements, unnecessary punctuation, or any additional commentary that does not affect the meaning.
    
#         For example, if given the input:
#         -------------------------------------------------
#         The original content of this audio is: 'Yesterday you were trembling for a health that is dear to you, today you fear for your own, tomorrow it will be anxiety about money, the day after tomorrow the diatribe of a slanderer, the day after that the misfortune of some friend, then the prevailing weather, then something that has been broken or lost, then a pleasure with which your conscience and your vertebral column rebel.
#         -------------------------------------------------
#         The expected cleaned output should be:
#         -------------------------------------------------
#         yesterday you were trembling for a health that is dear to you to day you fear for your own to morrow it will be anxiety about money the day after to morrow the diatribe of a slanderer the day after that the misfortune of some friend then the prevailing weather then something that has been broken or lost then a pleasure with which your conscience and your vertebral column reproach you again the course of public affairs
#         -------------------------------------------------
    
#         This prompt should be applicable in all cases—whether the task involves translation, processing multiple-choice options, or any similar scenario where extra expressions are present. Only output the cleaned text.
    
#         Now, please process the following text:
#         -------------------------------------------------
#         [INPUT]
#         -------------------------------------------------
#         The Output should only contain the cleaned text.
#         """

ASR_ALIGNMENT_PROMPT = """
    You are a **transcript-cleaning assistant**.  
    Your job is to output **only** the spoken content, exactly as it appears, after deleting any clearly redundant material.

    Guidelines  
    1. **Keep** every word that belongs to the transcript itself.  
    2. **Remove** headings, labels, metadata, timestamps, speaker tags, multiple-choice option letters, or any other text that is not part of what was said aloud.  
    3. Strip quotation marks or surplus punctuation **only** when this does not change the meaning. Do **not** rewrite, paraphrase, re-order, or shorten legitimate transcript content.  
    4. If you do **not** detect obvious redundancy, return the text unchanged.  
    5. Your response must contain the cleaned text **only**—no extra commentary.

    **Example**

    Input  
    -------------------------------------------------  
    The original content of this audio is: 'Yesterday you were trembling for a health that is dear to you, today you fear for your own, tomorrow it will be anxiety about money, the day after tomorrow the diatribe of a slanderer.'  
    -------------------------------------------------  

    Output  
    -------------------------------------------------  
    yesterday you were trembling for a health that is dear to you today you fear for your own tomorrow it will be anxiety about money the day after tomorrow the diatribe of a slanderer  
    -------------------------------------------------  

    Now clean the following text.  
    -------------------------------------------------  
    [INPUT]  
    -------------------------------------------------  

    (The output should consist **only** of the cleaned transcript.)
"""

class ASREvaluator(Evaluator):
    def __init__(self, model_name, meta_file, task, api, is_align=True, cache_dir='./cache'):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.task = task
        self.meta_file = meta_file
        self.client = CLIENT_MAP[api]()
        self.is_align = is_align
        self.cache_file = []
        self.metric = wer_metric
        self.align_prompt = ASR_ALIGNMENT_PROMPT

    def evaluate(self, data):
        print(f"Processing evaluation for model '{self.model_name}' on evaluator: '{self.task}'")
        scores_origin = None
        preds = []
        targets = []
        original = []
        for item in tqdm(data, total=len(data)):
            if 'common_voice' in item['file']:
                continue
            if self.is_align:
                pred = self.align_text(item)
                item['aligned_text'] = pred
            else:
                pred = item['pred']
            preds.append(pred)
            original.append(item['pred'])
            targets.append(item['target'])
        scores = self.metric(preds, targets)
        if scores['wer'] > 0.1:
            asr_score = 0
        else:
            asr_score = 1 - scores['wer']
        if self.is_align:
            scores_origin = self.metric(original, targets)
        self.save_cache(data)
        return {
                'model': self.model_name,
                'meta_file': self.meta_file,
                'task': self.task,
                'score': scores,
                'origin_score': scores_origin,
                'task_score': asr_score,
            }



import json
import os
import time

from tqdm import tqdm
from src.evaluation.api import CLIENT_MAP
from src.evaluation.evaluators.evaluator import Evaluator
from src.evaluation.metrics.accuracy import accuracy_metric, accuracy_metric_with_llm


ACC_ALIGNMENT_PROMPT = """
    Your task is to extract the specific answer choice from the `pred` text that corresponds to one of the options presented *within* the `question` text.
    
    **Rules:**
    
    1.  **Identify Choices:** Examine the `question` text to determine the valid multiple-choice options. 
    2.  **Analyze Prediction (`pred`):** Carefully read the `pred` text. Look for: the best suit option of the question. If not match any choice, output "no answer"
    3.  **Validation:** The extracted answer *must* be one of the valid choices identified in Step 1 from the `question`.
    4.  **Format Output:** IF TARGET IS A WORD, OUTPUT THE WORD. IF TARGET IS ABCD, THEN ONLY OUTPUT ABCD
    5.  **No Answer:** If the `pred` text does not clearly identify one of the valid choices listed in the `question`, output "no answer".

    
    **Input:**
    `question`: [QUESTION]
    `pred`: [INPUT]
    `target`: [TARGET]
    
    **Output:**
    """

class AccuracyEvaluator(Evaluator):
    def __init__(self, model_name, meta_file, task, api, is_align=True, cache_dir='./cache',  metric_type='llm'):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.task = task
        self.meta_file = meta_file
        self.client = CLIENT_MAP[api]()
        self.is_align = is_align
        self.cache_file = []
        self.metric_type = metric_type
        if metric_type == 'count':
            self.metric = accuracy_metric
        elif metric_type == 'llm':
            self.metric = accuracy_metric_with_llm
        self.align_prompt = ACC_ALIGNMENT_PROMPT

    def evaluate(self, data):
        print(f"Processing evaluation for model '{self.model_name}' on task: '{self.task}'")
        preds = []
        targets = []
        if self.metric_type == 'llm':
            scores = self.metric(self.client, data)
        else:
            for item in tqdm(data, total=len(data)):
                if self.is_align:
                    pred = self.extract_answer(item)
                    item['aligned_text'] = pred
                else:
                    pred = item['pred']
                preds.append(pred)
                targets.append(item['target'])
            scores = self.metric(preds, targets)

        # save data to cache
        self.save_cache(data)
        return {
                'model': self.model_name,
                'meta_file': self.meta_file,
                'task': self.task,
                'score': scores,
            }
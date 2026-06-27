
import json
import os
import time

from tqdm import tqdm
from src.evaluation.api import CLIENT_MAP
from src.evaluation.evaluators.evaluator import Evaluator
from src.evaluation.evaluators.asr_evaluator import ANSWER_EXTRACT_PROMPT
from src.evaluation.metrics.task_success_rate import task_success_rate_metric


class TaskSuccessRateEvaluator(Evaluator):
    def __init__(self, model_name, meta_file, task, api, is_align=True, cache_dir='./cache'):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.task = task
        self.meta_file = meta_file
        self.client = CLIENT_MAP[api]()
        self.is_align = is_align
        self.cache_file = []
        self.metric = task_success_rate_metric
        self.align_prompt = ANSWER_EXTRACT_PROMPT

    def evaluate(self, data):
        print(f"Processing evaluation for model '{self.model_name}' on evaluator: '{self.task}'")
        scores_origin = None
        preds = []
        targets = []
        original = []
        for item in tqdm(data, total=len(data)):
            if self.is_align:
                pred = self.extract_answer(item)
                item['aligned_text'] = pred
            else:
                pred = item['pred']
            preds.append(pred)
            original.append(item['pred'])
            targets.append(item['target'])
        scores = self.metric(preds, targets)
        task_score = scores['task_success_rate']
        if self.is_align:
            scores_origin = self.metric(original, targets)
        self.save_cache(data)
        return {
                'model': self.model_name,
                'meta_file': self.meta_file,
                'task': self.task,
                'score': scores,
                'origin_score': scores_origin,
                'task_score': task_score,
            }

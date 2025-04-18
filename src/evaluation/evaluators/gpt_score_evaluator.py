

import json
import os
import time

from tqdm import tqdm
from src.evaluation.api import CLIENT_MAP
from src.evaluation.evaluators.evaluator import Evaluator
from src.evaluation.metrics.gpt_eval import gpt_content_score, gpt_empathy_score
from src.evaluation.metrics.accuracy import accuracy_metric


class GPTScoreEvaluator(Evaluator):
    def __init__(self, model_name, meta_file, task, api, is_align=True, cache_dir='./cache'):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.task = task
        self.meta_file = meta_file
        self.client = CLIENT_MAP[api]()
        self.metric = gpt_empathy_score if task == 'empathy_score' else gpt_content_score

    def evaluate(self, data):
        print(f"Processing evaluation for model '{self.model_name}' on evaluator: '{self.task}'")
        scores = self.metric(self.client, data)
        # save data to cache

        self.save_cache(data)
        return {
                'model': self.model_name,
                'meta_file': self.meta_file,
                'task': self.task,
                'score': scores,
            }

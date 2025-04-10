

import json
import os

from tqdm import tqdm
from src.evaluation.api import CLIENT_MAP
from src.evaluation.evaluators.evaluator import Evaluator
from src.evaluation.metrics.gpt_eval import gpt_content_score, gpt_empathy_score
from src.evaluation.metrics.accuracy import accuracy_metric


ACC_ALIGNMENT_PROMPT = """
        You are a text processing assistant. Your task is to clean the provided text by removing any extraneous, redundant, or non-essential expressions while preserving the core semantic content. This includes eliminating introductory statements, irrelevant formatting elements, unnecessary punctuation, or any additional commentary that does not affect the meaning.
    
        For example, if given the input:
        -------------------------------------------------
        The original content of this audio is: 'Yesterday you were trembling for a health that is dear to you, today you fear for your own, tomorrow it will be anxiety about money, the day after tomorrow the diatribe of a slanderer, the day after that the misfortune of some friend, then the prevailing weather, then something that has been broken or lost, then a pleasure with which your conscience and your vertebral column rebel.
        -------------------------------------------------
        The expected cleaned output should be:
        -------------------------------------------------
        yesterday you were trembling for a health that is dear to you to day you fear for your own to morrow it will be anxiety about money the day after to morrow the diatribe of a slanderer the day after that the misfortune of some friend then the prevailing weather then something that has been broken or lost then a pleasure with which your conscience and your vertebral column reproach you again the course of public affairs
        -------------------------------------------------
    
        This prompt should be applicable in all cases—whether the task involves translation, processing multiple-choice options, or any similar scenario where extra expressions are present. Only output the cleaned text.
    
        Now, please process the following text:
        -------------------------------------------------
        [INPUT]
        -------------------------------------------------
        The Output should only contain the cleaned text.
        """

class GPTScoreEvaluator(Evaluator):
    def __init__(self, model_name, meta_file, evaluator, api, is_align=True, cache_dir='./cache'):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.evaluator = evaluator
        self.meta_file = meta_file
        self.client = CLIENT_MAP[api]()
        self.metric = gpt_empathy_score if evaluator == 'empathy_score' else gpt_content_score

    def evaluate(self, data):
        print(f"Processing evaluation for model '{self.model_name}' on evaluator: '{self.evaluator}'")
        scores = self.metric(self.client, data)
        return {
                'model': self.model_name,
                'meta_file': self.meta_file,
                'task': self.evaluator,
                'score': scores,
            }

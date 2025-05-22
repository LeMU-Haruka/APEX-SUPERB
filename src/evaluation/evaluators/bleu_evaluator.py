import json
import os
import time

from tqdm import tqdm
from src.evaluation.api import CLIENT_MAP
from src.evaluation.evaluators.evaluator import Evaluator
from src.evaluation.metrics.bleu import blue_metric


BLEU_EXTRACT_PROMPT = """
            # Task
            Clean the model's predicted German translation (`pred` field). **Your final output must be only the cleaned text string, containing no other characters or explanation.**
            
            # Goal
            Keep the main predicted text block from `pred`, removing only obvious non-translational noise surrounding it. Do not verify language or quality.
            
            # Input
            - `question`: English source (for reference only).
            - `pred`: Model's predicted text.
            
            # Cleaning Rules
            1.  **Keep:** The **core text block** in `pred` that looks like the model's main output attempt (regardless of content, including potential English or errors).
            2.  **Only Remove** obvious **surrounding noise** that is **external** to the core text block:
                *   Introductory phrases/tags (e.g., "German translation:", "Deutsch:").
                *   Explanatory sentences attached before or after (any language).
                *   Clearly appended metadata/timestamps.
            3.  **Core Principle:**
                *   **Do not edit/modify** the inside of the core text block.
                *   **Only remove absolutely certain surrounding noise.** If in doubt, **keep** it.
                *   If distinguishing is impossible or the entire `pred` is noise/explanation, **keep `pred` as is**.
            
            # Output Format (!!! Must Follow Strictly !!!)
            **Directly output** the cleaned text string (or the original `pred` string if kept unchanged by Rule 3).
            **Do NOT output** any steps, explanations, labels, code block markers, or any text other than the cleaned result itself. **The response must be the result string directly.**
            
            # Examples (For understanding rules only, do not mimic "Expected Output" label in your response)
            Example Input 1:
            Question: Hello world
            Prediction: German translation: Hallo Welt
            Expected Output: Hallo Welt
            
            Example Input 2:
            Question: as she awaited her guests...
            Prediction: Sie erwartete ihre Gäste und überprüfte das Tablett mit both satisfaction and disquietude... A selection of guests...
            Expected Output: Sie erwartete ihre Gäste und überprüfte das Tablett mit both satisfaction and disquietude... A selection of guests...
            
            Example Input 3:
            Question: Some English text.
            Prediction: The model failed to translate this segment.
            Expected Output: The model failed to translate this segment.
            
            # Now, strictly following the output format requirement, process the input below:
            Question: [QUESTION]
            Prediction: '[INPUT]'
            Expected Output(**Only output filted text, DO NOT ADD ANY EXPLANATION**):
        """

class BleuEvaluator(Evaluator):
    def __init__(self, model_name, meta_file, task, api, is_align=True, cache_dir='./cache'):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.task = task
        self.meta_file = meta_file
        self.client = CLIENT_MAP[api]()
        self.is_align = is_align
        self.cache_file = []
        self.metric = blue_metric
        self.align_prompt = BLEU_EXTRACT_PROMPT


    def evaluate(self, data):
        print(f"Processing evaluation for model '{self.model_name}' on evaluator: '{self.task}'")
        preds = []
        targets = []
        for item in tqdm(data, total=len(data)):
            if self.is_align:
                pred = self.extract_answer(item)
                item['aligned_text'] = pred
            else:
                pred = item['pred']
            preds.append(pred)
            targets.append(item['target'])
        scores = self.metric(preds, targets)
        task_score = scores['bleu_score']
        self.save_cache(data)
        return {
                'model': self.model_name,
                'meta_file': self.meta_file,
                'task': self.task,
                'score': scores,
                'task_score': task_score,
            }

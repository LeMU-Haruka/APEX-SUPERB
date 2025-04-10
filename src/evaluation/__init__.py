


from src.evaluation.evaluators.accuracy_evaluator import AccuracyEvaluator
from src.evaluation.evaluators.asr_evaluator import ASREvaluator
from src.evaluation.evaluators.bleu_evaluator import BleuEvaluator
from src.evaluation.evaluators.gpt_score_evaluator import GPTScoreEvaluator


EVALUATOR_MAP = {
    'asr': ASREvaluator,
    'st': BleuEvaluator,
    'accuracy': AccuracyEvaluator,
    'empathy_score': GPTScoreEvaluator,
    'alpaca_score': GPTScoreEvaluator,
}
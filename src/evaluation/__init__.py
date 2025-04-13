


from src.evaluation.evaluators.accuracy_evaluator import AccuracyEvaluator
from src.evaluation.evaluators.asr_evaluator import ASREvaluator
from src.evaluation.evaluators.bleu_evaluator import BleuEvaluator
from src.evaluation.evaluators.gpt_score_evaluator import GPTScoreEvaluator
from src.evaluation.evaluators.ifeval_evaluator import IfevalEvaluator


EVALUATOR_MAP = {
    'asr': ASREvaluator, # asr_librispeech, asr_commonvoice, librispeech_noise, librispeech_emotion, librispeech_speed, librispeech_multispeaker, text_instruct_asr, speech_instruct_asr
    'st': BleuEvaluator, # text_instruct_st
    'accuracy': AccuracyEvaluator, # animal_classification, sound_classification, dialogue_ser, emotion_recognition, speaker_role, mmau, mmlu, gsm8k
    'empathy_score': GPTScoreEvaluator, # alpaca_empathy, 
    'alpaca_score': GPTScoreEvaluator,  # alpaca_eval
    'ifEval': IfevalEvaluator, # ifeval
}
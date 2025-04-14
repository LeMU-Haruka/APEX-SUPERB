


from src.evaluation.evaluators.accuracy_evaluator import AccuracyEvaluator
from src.evaluation.evaluators.asr_evaluator import ASREvaluator
from src.evaluation.evaluators.bleu_evaluator import BleuEvaluator
from src.evaluation.evaluators.gpt_score_evaluator import GPTScoreEvaluator
from src.evaluation.evaluators.ifeval_evaluator import IfevalEvaluator


ASR_TASKS = [
    'asr_librispeech',
    'asr_commonvoice',
    'librispeech_noise',
    'librispeech_emotion',
    'librispeech_speed',
    'librispeech_multispeaker',
    'text_instruct_asr',
    'speech_instruct_asr'
]

ST_TASKS = [
    'text_instruct_st'
]

CLASSIFICATION_TASKS = [
    'animal_classification',
    'sound_classification',
    'dialogue_ser',
    'emotion_recognition',
    'speaker_role',
    'mmau',
    'mmlu',
    'gsm8k'
]

GPT_SCORE_TASKS = [
    'alpaca_empathy',
    'alpaca_eval'
]

IF_EVAL_TASKS = [
    'ifeval'
]


EVALUATOR_MAP = {
    'asr': ASREvaluator, # asr_librispeech, asr_commonvoice, librispeech_noise, librispeech_emotion, librispeech_speed, librispeech_multispeaker, text_instruct_asr, speech_instruct_asr
    'st': BleuEvaluator, # text_instruct_st
    'accuracy': AccuracyEvaluator, # animal_classification, sound_classification, dialogue_ser, emotion_recognition, speaker_role, mmau, mmlu, gsm8k
    'score': GPTScoreEvaluator, # alpaca_empathy, 
    'ifEval': IfevalEvaluator, # ifeval
}

# 根据上面的EVALUATOR_MAP和后面的注释，按照注释内容映射到指定的evaluator

def load_evaluator(evaluator_type, model_name, result_file, api, is_align=False):
    if evaluator_type in ASR_TASKS:
        evaluator_type = 'asr'
    elif evaluator_type in ST_TASKS:
        evaluator_type = 'st'
    elif evaluator_type in CLASSIFICATION_TASKS:
        evaluator_type = 'accuracy'
    elif evaluator_type in GPT_SCORE_TASKS:
        evaluator_type = 'score'
    elif evaluator_type in IF_EVAL_TASKS:
        evaluator_type = 'ifEval'
    evaluator = EVALUATOR_MAP[evaluator_type](model_name, result_file, evaluator_type, api, is_align)
    return evaluator
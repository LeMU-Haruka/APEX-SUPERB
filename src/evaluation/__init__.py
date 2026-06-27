


from src.evaluation.evaluators.accuracy_evaluator import AccuracyEvaluator
from src.evaluation.evaluators.asr_evaluator import ASREvaluator
from src.evaluation.evaluators.bleu_evaluator import BleuEvaluator
from src.evaluation.evaluators.gpt_score_evaluator import GPTScoreEvaluator
from src.evaluation.evaluators.ifeval_evaluator import IfevalEvaluator
from src.evaluation.evaluators.task_success_rate_evaluator import TaskSuccessRateEvaluator


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
    'mmlu_w_choice',
    'gsm8k',
    'gsm8k_fewshot_1',
    'gsm8k_fewshot_2',
    'gsm8k_fewshot_4',
    'gsm8k_fewshot_8',
]

GPT_SCORE_TASKS = [
    'alpaca_empathy',
    'alpaca_eval'
]

IF_EVAL_TASKS = [
    'ifeval'
]

TASK_SUCCESS_RATE_TASKS = [
    'instruction_robustness_asr',
    'instruction_robustness_asr_s'
]


EVALUATOR_MAP = {
    'asr': ASREvaluator, # asr_librispeech, asr_commonvoice, librispeech_noise, librispeech_emotion, librispeech_speed, librispeech_multispeaker, text_instruct_asr, speech_instruct_asr
    'st': BleuEvaluator, # text_instruct_st
    'accuracy': AccuracyEvaluator, # animal_classification, sound_classification, dialogue_ser, emotion_recognition, speaker_role, mmau, mmlu, gsm8k
    'score': GPTScoreEvaluator, # alpaca_empathy, 
    'ifeval': IfevalEvaluator, # ifeval
    'task_success_rate': TaskSuccessRateEvaluator, # instruction_robustness_asr, instruction_robustness_asr_s
}

def load_evaluator(task, model_name, result_file, api, is_align=False):
    if task in ASR_TASKS:
        evaluator = 'asr'
    elif task in ST_TASKS:
        evaluator = 'st'
    elif task in CLASSIFICATION_TASKS:
        evaluator = 'accuracy'
    elif task in GPT_SCORE_TASKS:
        evaluator = 'score'
    elif task in IF_EVAL_TASKS:
        evaluator = 'ifeval'
    elif task in TASK_SUCCESS_RATE_TASKS:
        evaluator = 'task_success_rate'
    else:
        supported_tasks = (
            ASR_TASKS
            + ST_TASKS
            + CLASSIFICATION_TASKS
            + GPT_SCORE_TASKS
            + IF_EVAL_TASKS
            + TASK_SUCCESS_RATE_TASKS
        )
        raise ValueError(f"Unsupported task '{task}'. Supported tasks: {supported_tasks}")
    evaluator = EVALUATOR_MAP[evaluator](model_name, result_file, task, api, is_align)
    return evaluator

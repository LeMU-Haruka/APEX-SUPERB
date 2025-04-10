from .base_dataset import BaseDataset, ChoiceDataset, HFDataset

# BASE_TASK_CONFIG_LIST = []

datasets_map = {
    'mmlu': BaseDataset,
    'alpaca_eval': BaseDataset,
    'asr_librispeech': BaseDataset,
    'asr_commonvoice': BaseDataset,
    'gsm8k': BaseDataset,
    'emotion': HFDataset,
    'dialogue_ser': HFDataset,
    'AED': HFDataset,
    'librispeech_multispeaker': BaseDataset,
    'librispeech_emotion': BaseDataset,
    'text_instruct_asr': BaseDataset,
    'text_instruct_st': BaseDataset,
    'empathy': BaseDataset,
    'speaker_role': BaseDataset,
    'librispeech_speed': BaseDataset,
    'ifeval': BaseDataset,
    'fewshot_gsm8k_1': BaseDataset,
    'fewshot_gsm8k_2': BaseDataset,
    'fewshot_gsm8k_4': BaseDataset,
    'fewshot_gsm8k_8': BaseDataset,
    'librispeech_noise': BaseDataset,
    'mmau': HFDataset,
}


DATASET_PATH_CONFIG = {
    'asr_librispeech': '/userhome/datasets/SGBench/librispeech',
    'asr_commonvoice': '/userhome/datasets/SGBench/commonvoice17_1000',
    'librispeech_multispeaker': '/userhome/datasets/SGBench/librispeech_multispeaker',
    'librispeech_emotion': '/userhome/datasets/SGBench/librispeech_emotion',
    'alpaca_eval': '/userhome/datasets/SGBench/alpacaeval_tts',
    'mmlu': '/userhome/datasets/SGBench/mmlu_tts',
    'nutshell': '/userhome/datasets/nutshell',
    'gsm8k': '/userhome/datasets/SGBench/gsm8k_tts',
    'emotion': '/userhome/datasets/SGBench/ser',
    'dialogue_ser': '/userhome/datasets/SGBench/dialogue_ser',
    'text_instruct_st': '/userhome/datasets/SGBench/text_instruct_st',
    'text_instruct_asr': '/userhome/datasets/SGBench/text_instruct_asr',
    'empathy': '/userhome/datasets/SGBench/alpaca_eval_emotion',
    'speaker_role': '/userhome/datasets/SGBench/speaker_role_tts',
    'librispeech_speed': '/userhome/datasets/SGBench/librispeech_speed',
    'ifeval': '/userhome/datasets/SGBench/IFEval',
    'fewshot_gsm8k_1': '/userhome/datasets/SGBench/gsm8k_tts',
    'fewshot_gsm8k_2': '/userhome/datasets/SGBench/gsm8k_tts',
    'fewshot_gsm8k_4': '/userhome/datasets/SGBench/gsm8k_tts',
    'fewshot_gsm8k_8': '/userhome/datasets/SGBench/gsm8k_tts',
    'librispeech_noise': '/userhome/datasets/SGBench/librispeech_noise',
    'mmau': '/userhome/datasets/SGBench/mmau'
}
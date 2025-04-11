import os
from .base_dataset import BaseDataset, ChoiceDataset, HFDataset

# BASE_TASK_CONFIG_LIST = []

datasets_map = {
    ###  foundation tasks ###
    # Speech
    'asr_librispeech': BaseDataset,
    'asr_commonvoice': BaseDataset,
    'emotion': HFDataset,
    'dialogue_ser': HFDataset,
    # sound
    'sound_classification': HFDataset,
    'animal_classification': HFDataset,
    ### Instruction tasks ###
    # multi-instruct
    'text_instruct_asr': BaseDataset,
    'text_instruct_st': BaseDataset,

    # speech instruction
    # TODO
    # IFEval
    'ifeval': BaseDataset,

    # Text Few shot instruction
    'fewshot_gsm8k_1': BaseDataset,
    'fewshot_gsm8k_2': BaseDataset,
    'fewshot_gsm8k_4': BaseDataset,
    'fewshot_gsm8k_8': BaseDataset,

    # Speech few shot instruction
    # TODO

    ### Input Adaptability tasks ###
    # Noisy robustness
    'librispeech_noise': BaseDataset,

    # Style speech adaptability
    'librispeech_multispeaker': BaseDataset,
    'librispeech_emotion': BaseDataset,
    'librispeech_speed': BaseDataset,

    ### Chat and Reasoning tasks ###
    # Knowledge
    'mmlu': BaseDataset,

    # Math reasoning 
    'gsm8k': BaseDataset,

    # content reasoning
    'speaker_role': BaseDataset,

    # audio reasoning
    'mmau': HFDataset,

    # dialogue
    'alpaca_eval': BaseDataset,
    'empathy': BaseDataset,

}

# 将DATASET_PATH_CONFIG按 上面的格式重新排列
# 方便后续使用

DATASET_ROOT = '/userhome/datasets/SGBench'

DATASET_PATH_CONFIG = {
    ###  foundation tasks ###
    # Speech
    'asr_librispeech': os.path.join(DATASET_ROOT, 'librispeech'),
    'asr_commonvoice': os.path.join(DATASET_ROOT, 'commonvoice17_1000'),
    'emotion': os.path.join(DATASET_ROOT, 'ser'),
    'dialogue_ser': os.path.join(DATASET_ROOT, 'dialogue_ser'),
    # sound
    'sound_classification': os.path.join(DATASET_ROOT, 'sound_classification'),
    'animal_classification': os.path.join(DATASET_ROOT, 'animal_classification'),
    ### Instruction tasks ###
    # multi-instruct
    'text_instruct_asr': os.path.join(DATASET_ROOT, 'text_instruct_asr'),
    'text_instruct_st': os.path.join(DATASET_ROOT, 'text_instruct_st'),
    # speech instruction
    # TODO

    # IFEval
    'ifeval': os.path.join(DATASET_ROOT, 'IFEval'),
    # Text Few shot instruction
    'fewshot_gsm8k_1': os.path.join(DATASET_ROOT, 'gsm8k_tts'),
    'fewshot_gsm8k_2': os.path.join(DATASET_ROOT, 'gsm8k_tts'),
    'fewshot_gsm8k_4': os.path.join(DATASET_ROOT, 'gsm8k_tts'),
    'fewshot_gsm8k_8': os.path.join(DATASET_ROOT, 'gsm8k_tts'),
    # Speech few shot instruction
    # TODO
    ### Input Adaptability tasks ###
    # Noisy robustness
    'librispeech_noise': os.path.join(DATASET_ROOT, 'librispeech_noise'),
    # Style speech adaptability
    'librispeech_multispeaker': os.path.join(DATASET_ROOT, 'librispeech_multispeaker'),
    'librispeech_emotion': os.path.join(DATASET_ROOT, 'librispeech_emotion'),
    'librispeech_speed': os.path.join(DATASET_ROOT, 'librispeech_speed'),
    ### Chat and Reasoning tasks ###
    # Knowledge
    'mmlu': os.path.join(DATASET_ROOT, 'mmlu_tts'),
    # Math reasoning
    'gsm8k': os.path.join(DATASET_ROOT, 'gsm8k_tts'),
    # content reasoning
    'speaker_role': os.path.join(DATASET_ROOT, 'speaker_role_tts'),
    # audio reasoning
    'mmau': os.path.join(DATASET_ROOT, 'mmau'),
    # dialogue
    'alpaca_eval': os.path.join(DATASET_ROOT, 'alpacaeval_tts'),
    'empathy': os.path.join(DATASET_ROOT, 'alpaca_eval_emotion'),
    'nutshell': os.path.join(DATASET_ROOT, 'nutshell'),

}

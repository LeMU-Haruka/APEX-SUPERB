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


DATASET_PATH_CONFIG = {
    ###  foundation tasks ###
    # Speech
    'asr_librispeech': '/userhome/datasets/SGBench/librispeech',
    'asr_commonvoice': '/userhome/datasets/SGBench/commonvoice17_1000',
    'emotion': '/userhome/datasets/SGBench/ser',
    'dialogue_ser': '/userhome/datasets/SGBench/dialogue_ser',
    # sound
    'sound_classification': '/userhome/datasets/SGBench/sound_classification',
    'animal_classification': '/userhome/datasets/SGBench/animal_classification',

    ### Instruction tasks ###
    # multi-instruct
    'text_instruct_asr': '/userhome/datasets/SGBench/text_instruct_asr',
    'text_instruct_st': '/userhome/datasets/SGBench/text_instruct_st',

    # speech instruction
    # TODO
    # IFEval
    'ifeval': '/userhome/datasets/SGBench/IFEval',

    # Text Few shot instruction
    'fewshot_gsm8k_1': '/userhome/datasets/SGBench/gsm8k_tts',
    'fewshot_gsm8k_2': '/userhome/datasets/SGBench/gsm8k_tts',
    'fewshot_gsm8k_4': '/userhome/datasets/SGBench/gsm8k_tts',
    'fewshot_gsm8k_8': '/userhome/datasets/SGBench/gsm8k_tts',

    # Speech few shot instruction
    # TODO

    ### Input Adaptability tasks ###
    # Noisy robustness
    'librispeech_noise': '/userhome/datasets/SGBench/librispeech_noise',

    # Style speech adaptability
    'librispeech_multispeaker': '/userhome/datasets/SGBench/librispeech_multispeaker',
    'librispeech_emotion': '/userhome/datasets/SGBench/librispeech_emotion',
    'librispeech_speed': '/userhome/datasets/SGBench/librispeech_speed',

    ### Chat and Reasoning tasks ###
    # Knowledge
    'mmlu': '/userhome/datasets/SGBench/mmlu_tts',

    # Math reasoning
    'gsm8k': '/userhome/datasets/SGBench/gsm8k_tts',

    # content reasoning
    'speaker_role': '/userhome/datasets/SGBench/speaker_role_tts',

    # audio reasoning
    'mmau': '/userhome/datasets/SGBench/mmau',

    # dialogue
    'alpaca_eval': '/userhome/datasets/SGBench/alpacaeval_tts',
    'empathy': '/userhome/datasets/SGBench/alpaca_eval_emotion',
    'nutshell': '/userhome/datasets/SGBench/nutshell',
    # Note: The key 'nutshell' from the original DATASET_PATH_CONFIG is omitted
    # as it's not present in the datasets_map structure provided.

}
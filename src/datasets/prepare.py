import os
from datasets import load_dataset


datasets = [
    # basic tasks
    "asr_commonvoice",
    "asr_librispeech",

    "dialogue_ser",
    "emotion_recognition",

    "animal_classification",
    "sound_classification",

    # instruction tasks
    "text_multi_instruction_st",
    "text_multi_instruction_ASR",
    "speech_multi_instruction_asr",
    "ifeval",

    # input robustness tasks
    "librispeech_noise",
    "librispeech_emotion",
    "librispeech_speed",
    "librispeech_multispeaker",

    
    # QA and reasoning tasks
    "gsm8k",
    "alpaca_empathy",
    "mmlu",
    "alpaca_eval",
    "speaker_role",
    "mmau"
]

output_dir = 'local_datasets'
os.makedirs(output_dir, exist_ok=True)



for task in datasets:
    if os.path.exists(os.path.join(output_dir, task)):
        print(f'{task} already exists, skip download')
        continue
    repo = f'LeMUHaruka/{task}'
    dataset = load_dataset(repo, split='test')

    dataset.save_to_disk(os.path.join(output_dir, task))
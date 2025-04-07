
import random
from src.datasets.task_prompts import GSM8K_FEWSHOT_RATIONALE


def create_few_shot_prompt(num_shots):
    """
    Create a few-shot prompt for the GSM8K dataset.
    """
    rationales = GSM8K_FEWSHOT_RATIONALE
    prompt  = 'Here are some examples:\n'
    for i in range(num_shots):
        rationale = random.choice(rationales)
        rationale = rationale.replace('[num]', str(i + 1))
        prompt += rationale + '\n'
    prompt += 'Now, follow the same process to solve the math question in the speech:\n'
    return prompt


TASK_PROMPTS = {
    'asr_librispeech': 'Transcribe this speech into text.',
    'asr_commonvoice': 'Transcribe this speech into text.',
    'librispeech_emotion': 'Transcribe this speech into text.',
    'librispeech_multispeaker': 'Transcribe this speech into text.',
    'gsm8k': 'Answer the math question in the speech.',
    'role': 'Infer the role of the speaker in the speech.',
    'alpaca_eval': 'Answer the question of user in the speech.',
    'mmlu': 'Choose an answer from the following options: ',
    'long_asr': 'Transcribe the whole speech into text.',
    'nutshell': 'Summarize this academic report into a short abstract.',
    'emotion': 'Recognize and organize the emotional expressions in the spoken words. The answer could be anger, disgust, sadness, joy, neutral, surprise, or fear.',
    'empathy': 'Generate an empathetic response based on the content and emotions expressed in the speech.',
    'speaker_role': 'Based on the speech content choose the most suitable role of the speaker from following answer: [student, teacher, doctor, police, engineer]',
    'librispeech_speed': 'Transcribe this speech into text.',
    'fewshot_gsm8k_1': create_few_shot_prompt(1),
    'fewshot_gsm8k_2': create_few_shot_prompt(2),
    'fewshot_gsm8k_4': create_few_shot_prompt(4),
    'fewshot_gsm8k_8': create_few_shot_prompt(8),
    'librispeech_noise': 'Transcribe this speech into text.',
}



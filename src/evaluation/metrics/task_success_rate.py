import jiwer
from tqdm import tqdm


def task_success_rate_metric(preds, targets):
    WER = 0.
    INSERTION = 0.
    success = 0
    for p, t in tqdm(zip(preds, targets), total=len(preds), desc="Calculating Task Success Rate"):
        wer = calculate_wer(p, t)
        insertion = calculate_insertion(p, t)
        WER += wer
        INSERTION += insertion
        if wer <= 0.5 and insertion <= 4:
            success += 1
    WER /= len(preds)
    INSERTION /= len(preds)
    task_success_rate = success / len(preds)
    print(f"Average WER: {WER:.3f}, Average Insertion: {INSERTION:.3f}, Task Success Rate: {task_success_rate:.3f}")
    return {
        'wer': WER,
        'insertion': INSERTION,
        'success': success,
        'total': len(preds),
        'task_success_rate': task_success_rate
    }


def calculate_wer(predicted_sentence, ground_truth):
    transform = jiwer.Compose([
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.ToLowerCase()
    ])
    gt = transform(ground_truth)
    pred = transform(predicted_sentence)
    wer = jiwer.wer(gt, pred)
    return wer


def calculate_insertion(predicted_sentence, ground_truth):
    transform = jiwer.Compose([
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.ToLowerCase()
    ])
    gt = transform(ground_truth)
    pred = transform(predicted_sentence)
    output = jiwer.process_words(gt, pred)
    return output.insertions

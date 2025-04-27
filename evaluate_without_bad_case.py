from utils import load_result_files
import json
import argparse
from evaluate_ASR_generalization import calculate_wer, calculate_cer

def wer_metric(preds, targets):
    # 计算平均wer和cer
    WER = 0.
    CER = 0.
    N = len(preds)
    for p, t in zip(preds, targets):
        wer = calculate_wer(p, t)
        cer = calculate_cer(p, t)
        if wer >= 2:
            N = N - 1
        else:
            WER += wer
            CER += cer
    WER /= N
    CER /= N
    # print(f"Average WER: {WER:.3f}, Average CER: {CER:.3f}")
    # 返回dict格式
    return {
        'wer': round(WER * 100, 2),
        'cer': round(CER * 100, 2),
        'filter rate': round((len(preds) - N) / len(preds) * 100, 2),
    }


def WER(file):
    # print(f"Caculating WER for  file: '{file}'")
    with open(file, 'r') as f:
        data = json.load(f)
    preds_aligned = [item["aligned_text"] for item in data]
    preds_pred = [item["pred"] for item in data]
    targets = [item["target"] for item in data]
    scores_aligned = wer_metric(preds_aligned, targets)
    scores_pred = wer_metric(preds_pred, targets)
    
    scores = {
        'wer': min(scores_aligned['wer'], scores_pred['wer']),
        'cer': min(scores_aligned['cer'], scores_pred['cer']),
        'filter rate': scores_aligned['filter rate'] if scores_aligned['wer'] < scores_pred['wer'] else scores_pred['filter rate']
    }
    return scores

def main(args):

    files = load_result_files(args.input_dir)
    print(f"file_num: {len(files)}")

    from collections import defaultdict

    def nested_dict():
        return defaultdict(nested_dict)

    results = nested_dict()

    for file in files:
        _, _, _, dataset_name, instruction_class, filename = file.split('/')
        tmp = filename.split("_")[:-2]
        model_name = "_".join(tmp[:-1])
        instruction_type = tmp[-1]
        results[instruction_type][dataset_name][model_name][instruction_class] = WER(file)
    with open(f"../data/results_filtered.json", 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", '-i', type=str, default="../data/classified", help="输入JSON文件目录")
    args = parser.parse_args()
    
    main(args)


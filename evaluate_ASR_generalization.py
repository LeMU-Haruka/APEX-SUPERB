from utils import load_result_files
import json
import os
import argparse
import jiwer

def wer_metric(preds, targets):
    # 计算平均wer和cer
    WER = 0.
    CER = 0.
    for p, t in zip(preds, targets):
        wer = calculate_wer(p, t)
        cer = calculate_cer(p, t)
        WER += wer
        CER += cer
    WER /= len(preds)
    CER /= len(preds)
    print(f"Average WER: {WER:.3f}, Average CER: {CER:.3f}")
    # 返回dict格式
    return {
        'wer': WER,
        'cer': CER
    }


def calculate_wer(predicted_sentence, ground_truth):
    """
    计算单词错误率 (Word Error Rate - WER).

    Args:
        predicted_sentence (str):  ASR 模型预测的句子.
        ground_truth (str):  真实的句子 (ground truth).

    Returns:
        float:  WER 值 (0.0 表示完全正确, 1.0 表示完全错误).
    """

    transform = jiwer.Compose([
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.ToLowerCase()
    ])
    gt = transform(ground_truth)
    pred = transform(predicted_sentence)
    wer = jiwer.wer(gt, pred)
    return wer


def calculate_cer(predicted_sentence, ground_truth):
    """
    计算字符错误率 (Character Error Rate - CER).

    Args:
        predicted_sentence (str):  ASR 模型预测的句子.
        ground_truth (str):  真实的句子 (ground truth).

    Returns:
        float:  CER 值 (0.0 表示完全正确, 1.0 表示完全错误).
    """
    transform = jiwer.Compose([
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.ToLowerCase()
    ])
    gt = transform(ground_truth)
    pred = transform(predicted_sentence)
    cer = jiwer.cer(gt, pred)
    return cer

output_dir = "../data/eval_results"
os.makedirs(output_dir, exist_ok=True)

def WER(file, type):
    print(f"Caculating WER for  file: '{file}'")
    with open(file, 'r') as f:
        data = json.load(f)
    if type == "aligned":
        preds = [item["aligned_text"] for item in data]
    elif type == "pred":
        preds = [item["pred"] for item in data]
    else:
        raise KeyError
    targets = [item["target"] for item in data]
    scores = wer_metric(preds, targets)
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
        results[model_name][dataset_name][instruction_type][instruction_class] = WER(file, args.type)
        print(f"model_name: {model_name}, dataset_name: {dataset_name}, instruction_type: {instruction_type}, instruction_class: {instruction_class}")
    with open(f"../data/results_{args.type}.json", 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", '-i', type=str, default="../data/classified", help="输入JSON文件目录")
    parser.add_argument("--type", '-t', type=str, required=True, choices=["pred", "aligned"], help="输出目录")
    args = parser.parse_args()
    
    main(args)


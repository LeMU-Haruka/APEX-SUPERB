from evaluate_ASR_generalization import WER
import os
import json
from collections import defaultdict

def nested_dict():
    return defaultdict(nested_dict)

results = nested_dict()

clean_dir = "../data/clean"
for file in os.listdir(clean_dir):
    tmp = file.rstrip(".json").split("_")
    model_name = "_".join(tmp[:-2])
    dataset = tmp[-1]
    wer_pred = WER(os.path.join(clean_dir, file), "pred")
    wer_aligned = WER(os.path.join(clean_dir, file), "aligned")
    wer = min(wer_pred["wer"], wer_aligned["wer"])
    cer = min(wer_pred["cer"], wer_aligned["cer"])
    results[model_name][dataset]['wer'] = wer
    results[model_name][dataset]['cer'] = cer
    # print(f"model_name: {model_name}, dataset: {dataset}, wer: {wer}, cer: {cer}")

with open("../data/results_clean.json", 'w') as f:
    json.dump(results, f, indent=4)
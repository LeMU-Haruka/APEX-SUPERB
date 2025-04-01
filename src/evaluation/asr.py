import json
import re

import jiwer

from src.evaluation.evaluator import Evaluator


class AsrEvaluator(Evaluator):



    def calculate_wer(self, predicted_sentence, ground_truth):
        """
        计算词错误率 (Word Error Rate - WER).

        Args:
            predicted_sentence (str):  ASR 模型预测的句子.
            ground_truth (str):  真实的句子 (ground truth).

        Returns:
            float:  WER 值 (0.0 表示完全正确, 1.0 表示完全错误).
        """
        predicted_sentence = predicted_sentence.replace(":'", ",'").replace(".'", ".'").replace("-", " ")
        ground_truth = ground_truth.replace(":'", ",'").replace(".'", ".'").replace("-", " ")
        error = jiwer.wer(ground_truth, predicted_sentence)
        return error


    def calculate_cer(self, predicted_sentence, ground_truth):
        """
        计算字符错误率 (Character Error Rate - CER).

        Args:
            predicted_sentence (str):  ASR 模型预测的句子.
            ground_truth (str):  真实的句子 (ground truth).

        Returns:
            float:  CER 值 (0.0 表示完全正确, 1.0 表示完全错误).
        """
        error = jiwer.cer(ground_truth, predicted_sentence)
        return error


    def evaluate(self, data, task='asr'):
        """
        评估 ASR 模型的性能，计算 WER 和 CER.

        Args:
            data (obj):  包含 'predicted' 和 'ground_truth' 字段的对象.

        Returns:
            dict:  包含 WER 和 CER 的字典.  {'wer': wer_value, 'cer': cer_value}
        """
        total_wer = 0.
        total_cer = 0.
        for item in data:
            predicted = item['pred']
            ground_truth = item['target']
            predicted_sentence = " ".join(predicted.strip().split())
            ground_truth = " ".join(ground_truth.strip().split())

            match = re.search(r"'(.*?)'", predicted_sentence)  # 非贪婪匹配
            if match:
                predicted_sentence = match.group(1)

            predicted_sentence = predicted_sentence.replace(":", " ").replace(",", " ").replace(".", " ").lower()

            wer_value = self.calculate_wer(predicted_sentence, ground_truth)
            cer_value = self.calculate_wer(predicted_sentence, ground_truth)

            total_wer += wer_value
            total_cer += cer_value
        wer = total_wer / len(data)
        cer = total_cer / len(data)
        return {
            'wer': wer,
            'cer': cer,
        }

#
# if __name__ == '__main__':
#     # 示例用法
#     predicted = "this is an asr test sentenc"  # 故意引入一些错误
#     ground_truth = "this is an ASR test sentence"
#
#     results = evaluate_asr(predicted, ground_truth)
#     print(f"WER: {results['wer']:.4f}")  # 格式化输出，保留四位小数
#     print(f"CER: {results['cer']:.4f}")
#
#     predicted = "你好世界"
#     ground_truth = "你好 世界"
#     results = evaluate_asr(predicted, ground_truth)
#     print(f"WER: {results['wer']:.4f}")
#     print(f"CER: {results['cer']:.4f}")
#
#     predicted = "完全没有错误"
#     ground_truth = "完全没有错误"
#     results = evaluate_asr(predicted, ground_truth)
#     print(f"WER: {results['wer']:.4f}")
#     print(f"CER: {results['cer']:.4f}")
#
#     predicted = "是"  #空字符串测试
#     ground_truth = "这是一个句子"
#     results = evaluate_asr(predicted, ground_truth)
#     print(f"WER: {results['wer']:.4f}")
#     print(f"CER: {results['cer']:.4f}")

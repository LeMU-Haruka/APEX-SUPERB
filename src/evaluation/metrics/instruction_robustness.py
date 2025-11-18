import re
from collections import Counter
from bert_score import score
import jiwer
from tqdm import tqdm

model_path = '/extrahome0/xiejingran/models/roberta-large'

def normalize_text(text: str) -> str:
    transform = jiwer.Compose([
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.ToLowerCase()
    ])
    text = transform(text)
    return text

def word_f1_score(pred: str, ref: str) -> float:
    """词级别 F1，用于短句（len(ref) < 5）"""
    pred_words = pred.split()
    ref_words = ref.split()
    if not ref_words:
        return 1.0 if not pred_words else 0.0
    if not pred_words:
        return 0.0

    pred_counter = Counter(pred_words)
    ref_counter = Counter(ref_words)
    overlap = sum((pred_counter & ref_counter).values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_words)
    recall = overlap / len(ref_words)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def adaptive_similarity(
    pred: str,
    ref: str,
    short_threshold: int = 5,
    lang: str = "en",
    model_type: str = "roberta-large",
    verbose: bool = False
) -> float:
    """
    自适应句子相似度评估：
      - 若 reference 词数 < short_threshold (默认5)，使用 Word F1（鲁棒、容错）
      - 否则，使用 BERTScore F1（捕捉语义）
    
    Args:
        pred (str): 模型输出（可能含指令模板）
        ref (str): 参考文本（如 LibriSpeech ground truth）
        short_threshold (int): 切换阈值，默认5
        lang, model_type: BERTScore 参数

    Returns:
        float: 相似度分数 ∈ [0, 1]
    """
    pred = normalize_text(pred)
    ref = normalize_text(ref)

    ref_words = ref.split()
    if len(ref_words) < short_threshold:
        return word_f1_score(pred, ref)
    else:
        # 使用 BERTScore（自动处理 batch）
        _, _, f1 = score(
            [pred],
            [ref],
            lang=lang,
            model_type=model_path,
            num_layers=24, 
            verbose=verbose,
            device="cuda"  # 可改为 "cuda" 加速
        )
        return f1.item()


def instruction_robustness_metric(preds, targets):
    success = 0
    cache = []
    for p, t in tqdm(zip(preds, targets), total=len(preds), desc="Calculating Instruction Robustness"):
        similarity = adaptive_similarity(p, t)
        cache.append(similarity)
        if similarity > 0.7:
            success += 1
    print(f"Success rate is {success/len(preds):.3f}")
    # 返回dict格式
    return {
        'success_rate': success / len(preds),
        'similarity_scores': cache
    }
    

# pred = "The sentence spoken in the audio is as follows: 'Dorcas, in her strange way, was moved.'"
# target = "dorcas in her strange way was moved"


# sim = adaptive_similarity(pred, target)
# print(f"Adaptive Similarity: {sim:.4f}")
from sacrebleu.metrics import BLEU
import sacrebleu # 导入顶层包以获取版本信息等

def blue_metric(pred_list, target_list):
    """
    使用 sacrebleu.metrics.BLEU 类计算 BLEU 分数。

    Args:
        pred_list (list[str]): 预测句子列表。
        target_list (list[str]): 目标（参考）句子列表。一一对应。
        lowercase (bool): 是否在计算前将文本转为小写 (推荐 True)。
        tokenize (str): 使用哪种 sacrebleu 内置分词器。
                        常用: '13a' (WMT 默认之一), 'intl' (国际文本), 'none' (不分词)。
        max_ngram (int): 计算 BLEU 时考虑的最大 n-gram (默认 4 for BLEU-4)。

    Returns:
        float: 计算得到的 BLEU 分数 (范围 0 到 1)。
               返回 None 如果输入列表为空或长度不匹配。
    """
    if not pred_list or not target_list or len(pred_list) != len(target_list):
        print("Error: Input lists are empty or have mismatched lengths.")
        return None

    # sacrebleu 要求目标是 list of lists of strings
    # [[ref_sent1_alt1, ref_sent1_alt2], [ref_sent2_alt1], ...]
    # 因为我们只有一个参考，格式是: [[ref1], [ref2], ...]
    formatted_targets = [[tgt] for tgt in target_list]

    # 1. 实例化 BLEU 度量对象
    # 可以配置多个参数，这里使用常用设置
    bleu_metric = BLEU()

    # 2. 计算分数
    # .score() 方法接受预测列表和格式化后的参考列表
    score_result = bleu_metric.corpus_score(pred_list, formatted_targets)

    # score_result 是一个包含多个字段的对象，例如:
    # score_result.score: BLEU score (0-100)
    # score_result.counts: n-gram 匹配数
    # score_result.totals: n-gram 总数
    # score_result.precisions: 各 n-gram 精度 (0-100)
    # score_result.bp: Brevity Penalty
    # score_result.sys_len: 系统输出总长度
    # score_result.ref_len: 参考输出总长度

    # 返回dict格式
    return {
        'bleu_score': score_result.score,
        'bleu_score_0_1': score_result.score/100,
        'counts': score_result.counts,
        'totals': score_result.totals,
        'precisions': score_result.precisions,
        'bp': score_result.bp,
        'sys_len': score_result.sys_len,
        'ref_len': score_result.ref_len
    }
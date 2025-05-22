from sacrebleu import corpus_bleu

def blue_metric(pred_list, target_list):
    if not pred_list or not target_list or len(pred_list) != len(target_list):
        print("Error: Input lists are empty or have mismatched lengths.")
        return None

    formatted_targets = [target_list]


    score_result = corpus_bleu(
        pred_list, 
        formatted_targets,
        tokenize="13a",         
        lowercase=False,
        use_effective_order=True
        )

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

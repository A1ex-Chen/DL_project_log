def summarization_metrics(pred: EvalPrediction) ->Dict:
    pred_str, label_str = decode_pred(pred)
    rouge: Dict = calculate_rouge(pred_str, label_str)
    summ_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
    rouge.update({'gen_len': summ_len})
    return rouge

def translation_metrics(pred: EvalPrediction) ->Dict:
    pred_str, label_str = decode_pred(pred)
    bleu: Dict = calculate_bleu(pred_str, label_str)
    gen_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
    bleu.update({'gen_len': gen_len})
    return bleu

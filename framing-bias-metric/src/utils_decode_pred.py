def decode_pred(pred: EvalPrediction) ->Tuple[List[str], List[str]]:
    pred_str = tokenizer.batch_decode(pred.predictions, skip_special_tokens
        =True)
    label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True
        )
    pred_str = lmap(str.strip, pred_str)
    label_str = lmap(str.strip, label_str)
    return pred_str, label_str

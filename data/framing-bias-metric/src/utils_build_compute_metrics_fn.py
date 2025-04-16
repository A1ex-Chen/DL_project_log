def build_compute_metrics_fn(task_name: str, tokenizer: PreTrainedTokenizer
    ) ->Callable[[EvalPrediction], Dict]:

    def non_pad_len(tokens: np.ndarray) ->int:
        return np.count_nonzero(tokens != tokenizer.pad_token_id)

    def decode_pred(pred: EvalPrediction) ->Tuple[List[str], List[str]]:
        pred_str = tokenizer.batch_decode(pred.predictions,
            skip_special_tokens=True)
        label_str = tokenizer.batch_decode(pred.label_ids,
            skip_special_tokens=True)
        pred_str = lmap(str.strip, pred_str)
        label_str = lmap(str.strip, label_str)
        return pred_str, label_str

    def summarization_metrics(pred: EvalPrediction) ->Dict:
        pred_str, label_str = decode_pred(pred)
        rouge: Dict = calculate_rouge(pred_str, label_str)
        summ_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
        rouge.update({'gen_len': summ_len})
        return rouge

    def translation_metrics(pred: EvalPrediction) ->Dict:
        pred_str, label_str = decode_pred(pred)
        bleu: Dict = calculate_bleu(pred_str, label_str)
        gen_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
        bleu.update({'gen_len': gen_len})
        return bleu
    compute_metrics_fn = (summarization_metrics if 'summarization' in
        task_name else translation_metrics)
    return compute_metrics_fn

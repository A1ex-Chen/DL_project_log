def __call__(self, eval_preds: EvalPrediction) ->Dict[str, Any]:
    preds, targets = eval_preds
    logger.warning(
        f'preds shape: {preds.shape}. targets shape: {targets.shape}')
    preds = decode_generate_ids(self.tokenizer, preds)
    targets = decode_generate_ids(self.tokenizer, targets)
    assert len(preds) == len(targets)
    return self.calculate_metric(preds, targets)

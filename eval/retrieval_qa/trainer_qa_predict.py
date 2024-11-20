def predict(self, predict_dataset, predict_examples, ignore_keys=None,
    metric_key_prefix: str='test'):
    predict_dataloader = self.get_test_dataloader(predict_dataset)
    compute_metrics = self.compute_metrics
    self.compute_metrics = None
    eval_loop = (self.prediction_loop if self.args.
        use_legacy_prediction_loop else self.evaluation_loop)
    try:
        output = eval_loop(predict_dataloader, description='Prediction',
            prediction_loss_only=True if compute_metrics is None else None,
            ignore_keys=ignore_keys)
    finally:
        self.compute_metrics = compute_metrics
    if self.post_process_function is None or self.compute_metrics is None:
        return output
    predictions = self.post_process_function(predict_examples,
        predict_dataset, output.predictions, 'predict')
    metrics = self.compute_metrics(predictions)
    for key in list(metrics.keys()):
        if not key.startswith(f'{metric_key_prefix}_'):
            metrics[f'{metric_key_prefix}_{key}'] = metrics.pop(key)
    return PredictionOutput(predictions=predictions.predictions, label_ids=
        predictions.label_ids, metrics=metrics)

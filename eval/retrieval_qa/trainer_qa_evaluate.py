def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None,
    metric_key_prefix: str='eval'):
    eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
    eval_dataloader = self.get_eval_dataloader(eval_dataset)
    eval_examples = (self.eval_examples if eval_examples is None else
        eval_examples)
    compute_metrics = self.compute_metrics
    self.compute_metrics = None
    eval_loop = (self.prediction_loop if self.args.
        use_legacy_prediction_loop else self.evaluation_loop)
    try:
        output = eval_loop(eval_dataloader, description='Evaluation',
            prediction_loss_only=True if compute_metrics is None else None,
            ignore_keys=ignore_keys)
    finally:
        self.compute_metrics = compute_metrics
    if (self.post_process_function is not None and self.compute_metrics is not
        None):
        eval_preds = self.post_process_function(eval_examples, eval_dataset,
            output.predictions)
        metrics = self.compute_metrics(eval_preds)
        for key in list(metrics.keys()):
            if not key.startswith(f'{metric_key_prefix}_'):
                metrics[f'{metric_key_prefix}_{key}'] = metrics.pop(key)
        self.log(metrics)
    else:
        metrics = {}
    if self.args.tpu_metrics_debug or self.args.debug:
        xm.master_print(met.metrics_report())
    self.control = self.callback_handler.on_evaluate(self.args, self.state,
        self.control, metrics)
    return metrics

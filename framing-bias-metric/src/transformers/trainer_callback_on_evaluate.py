def on_evaluate(self, args, state, control, metrics, **kwargs):
    metric_to_check = args.metric_for_best_model
    if not metric_to_check.startswith('eval_'):
        metric_to_check = f'eval_{metric_to_check}'
    metric_value = metrics.get(metric_to_check)
    if metric_value is None:
        logger.warning(
            f'early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping is disabled'
            )
        return
    self.check_metric_value(args, state, control, metric_value)
    if self.early_stopping_patience_counter >= self.early_stopping_patience:
        control.should_training_stop = True

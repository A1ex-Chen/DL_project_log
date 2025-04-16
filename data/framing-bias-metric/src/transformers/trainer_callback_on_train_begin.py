def on_train_begin(self, args, state, control, **kwargs):
    assert args.load_best_model_at_end, 'EarlyStoppingCallback requires load_best_model_at_end = True'
    assert args.metric_for_best_model is not None, 'EarlyStoppingCallback requires metric_for_best_model is defined'
    assert args.evaluation_strategy != EvaluationStrategy.NO, 'EarlyStoppingCallback requires EvaluationStrategy of steps or epoch'

def on_log(self, args, state, control, logs=None, **kwargs):
    if args.evaluation_strategy == EvaluationStrategy.NO and 'loss' in logs:
        values = {'Training Loss': logs['loss']}
        values['Step'] = state.global_step
        self.training_tracker.write_line(values)

def on_train_begin(self, args, state, control, **kwargs):
    self.first_column = ('Epoch' if args.evaluation_strategy ==
        EvaluationStrategy.EPOCH else 'Step')
    self.training_loss = 0
    self.last_log = 0
    column_names = [self.first_column] + ['Training Loss']
    if args.evaluation_strategy != EvaluationStrategy.NO:
        column_names.append('Validation Loss')
    self.training_tracker = NotebookTrainingTracker(state.max_steps,
        column_names)

def on_epoch_end(self, args: TrainingArguments, state: TrainerState,
    control: TrainerControl, **kwargs):
    if args.evaluation_strategy == EvaluationStrategy.EPOCH:
        control.should_evaluate = True
        if args.load_best_model_at_end:
            control.should_save = True
    return control

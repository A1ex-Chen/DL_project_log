def on_step_begin(self, args: TrainingArguments, state: TrainerState,
    control: TrainerControl):
    control.should_log = False
    control.should_evaluate = False
    control.should_save = False
    return self.call_event('on_step_begin', args, state, control)

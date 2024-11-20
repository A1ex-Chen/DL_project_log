def on_save(self, args: TrainingArguments, state: TrainerState, control:
    TrainerControl):
    control.should_save = False
    return self.call_event('on_save', args, state, control)

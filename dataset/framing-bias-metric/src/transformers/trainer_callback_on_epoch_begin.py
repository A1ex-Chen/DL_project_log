def on_epoch_begin(self, args: TrainingArguments, state: TrainerState,
    control: TrainerControl):
    control.should_epoch_stop = False
    return self.call_event('on_epoch_begin', args, state, control)

def on_init_end(self, args: TrainingArguments, state: TrainerState, control:
    TrainerControl):
    return self.call_event('on_init_end', args, state, control)

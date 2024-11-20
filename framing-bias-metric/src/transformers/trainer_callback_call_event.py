def call_event(self, event, args, state, control, **kwargs):
    for callback in self.callbacks:
        result = getattr(callback, event)(args, state, control, model=self.
            model, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler,
            train_dataloader=self.train_dataloader, eval_dataloader=self.
            eval_dataloader, **kwargs)
        if result is not None:
            control = result
    return control

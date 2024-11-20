def on_prediction_step(self, args, state, control, eval_dataloader=None, **
    kwargs):
    if state.is_local_process_zero:
        if self.prediction_bar is None:
            self.prediction_bar = tqdm(total=len(eval_dataloader), leave=
                self.training_bar is None)
        self.prediction_bar.update(1)

def on_prediction_step(self, args, state, control, eval_dataloader=None, **
    kwargs):
    if self.prediction_bar is None:
        if self.training_tracker is not None:
            self.prediction_bar = self.training_tracker.add_child(len(
                eval_dataloader))
        else:
            self.prediction_bar = NotebookProgressBar(len(eval_dataloader))
        self.prediction_bar.update(1)
    else:
        self.prediction_bar.update(self.prediction_bar.value + 1)

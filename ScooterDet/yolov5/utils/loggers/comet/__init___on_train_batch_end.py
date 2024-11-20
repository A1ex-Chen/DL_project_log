def on_train_batch_end(self, log_dict, step):
    self.experiment.curr_step = step
    if self.log_batch_metrics and step % self.comet_log_batch_interval == 0:
        self.log_metrics(log_dict, step=step)
    return

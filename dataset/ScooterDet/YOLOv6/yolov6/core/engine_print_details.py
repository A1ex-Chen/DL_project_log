def print_details(self):
    if self.main_process:
        self.mean_loss = (self.mean_loss * self.step + self.loss_items) / (self
            .step + 1)
        self.pbar.set_description(('%10s' + ' %10.4g' + '%10.4g' * self.
            loss_num) % (f'{self.epoch}/{self.max_epoch - 1}', self.
            scheduler.get_last_lr()[0], *self.mean_loss))

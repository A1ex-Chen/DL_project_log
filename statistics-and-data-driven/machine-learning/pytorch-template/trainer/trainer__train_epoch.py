def _train_epoch(self, epoch):
    """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
    self.model.train()
    self.train_metrics.reset()
    for batch_idx, (data, target) in enumerate(self.data_loader):
        data, target = data.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
        self.train_metrics.update('loss', loss.item())
        for met in self.metric_ftns:
            self.train_metrics.update(met.__name__, met(output, target))
        if batch_idx % self.log_step == 0:
            self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                epoch, self._progress(batch_idx), loss.item()))
            self.writer.add_image('input', make_grid(data.cpu(), nrow=8,
                normalize=True))
        if batch_idx == self.len_epoch:
            break
    log = self.train_metrics.result()
    if self.do_validation:
        val_log = self._valid_epoch(epoch)
        log.update(**{('val_' + k): v for k, v in val_log.items()})
    if self.lr_scheduler is not None:
        self.lr_scheduler.step()
    return log

def _valid_epoch(self, epoch):
    """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
    self.model.eval()
    self.valid_metrics.reset()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(self.valid_data_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            self.writer.set_step((epoch - 1) * len(self.valid_data_loader) +
                batch_idx, 'valid')
            self.valid_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(output, target))
            self.writer.add_image('input', make_grid(data.cpu(), nrow=8,
                normalize=True))
    for name, p in self.model.named_parameters():
        self.writer.add_histogram(name, p, bins='auto')
    return self.valid_metrics.result()

def train(self):
    accum_iter = 0
    self.validate(0, accum_iter)
    for epoch in range(self.num_epochs):
        accum_iter = self.train_one_epoch(epoch, accum_iter)
        self.validate(epoch, accum_iter)
    self.logger_service.complete({'state_dict': self._create_state_dict()})
    self.writer.close()

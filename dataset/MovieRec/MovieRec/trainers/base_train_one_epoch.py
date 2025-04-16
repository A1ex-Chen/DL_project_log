def train_one_epoch(self, epoch, accum_iter):
    self.model.train()
    if self.args.enable_lr_schedule:
        self.lr_scheduler.step()
    average_meter_set = AverageMeterSet()
    tqdm_dataloader = tqdm(self.train_loader)
    for batch_idx, batch in enumerate(tqdm_dataloader):
        batch_size = batch[0].size(0)
        batch = [x.to(self.device) for x in batch]
        self.optimizer.zero_grad()
        loss = self.calculate_loss(batch)
        loss.backward()
        self.optimizer.step()
        average_meter_set.update('loss', loss.item())
        tqdm_dataloader.set_description('Epoch {}, loss {:.3f} '.format(
            epoch + 1, average_meter_set['loss'].avg))
        accum_iter += batch_size
        if self._needs_to_log(accum_iter):
            tqdm_dataloader.set_description('Logging to Tensorboard')
            log_data = {'state_dict': self._create_state_dict(), 'epoch': 
                epoch + 1, 'accum_iter': accum_iter}
            log_data.update(average_meter_set.averages())
            self.log_extra_train_info(log_data)
            self.logger_service.log_train(log_data)
    return accum_iter

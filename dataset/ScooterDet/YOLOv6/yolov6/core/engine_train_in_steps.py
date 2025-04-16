def train_in_steps(self, epoch_num, step_num):
    images, targets = self.prepro_data(self.batch_data, self.device)
    if self.write_trainbatch_tb and self.main_process and self.step == 0:
        self.plot_train_batch(images, targets)
        write_tbimg(self.tblogger, self.vis_train_batch, self.step + self.
            max_stepnum * self.epoch, type='train')
    with amp.autocast(enabled=self.device != 'cpu'):
        _, _, batch_height, batch_width = images.shape
        preds, s_featmaps = self.model(images)
        if self.args.distill:
            with torch.no_grad():
                t_preds, t_featmaps = self.teacher_model(images)
            temperature = self.args.temperature
            total_loss, loss_items = self.compute_loss_distill(preds,
                t_preds, s_featmaps, t_featmaps, targets, epoch_num, self.
                max_epoch, temperature, step_num, batch_height, batch_width)
        elif self.args.fuse_ab:
            total_loss, loss_items = self.compute_loss((preds[0], preds[3],
                preds[4]), targets, epoch_num, step_num, batch_height,
                batch_width)
            total_loss_ab, loss_items_ab = self.compute_loss_ab(preds[:3],
                targets, epoch_num, step_num, batch_height, batch_width)
            total_loss += total_loss_ab
            loss_items += loss_items_ab
        else:
            total_loss, loss_items = self.compute_loss(preds, targets,
                epoch_num, step_num, batch_height, batch_width)
        if self.rank != -1:
            total_loss *= self.world_size
    self.scaler.scale(total_loss).backward()
    self.loss_items = loss_items
    self.update_optimizer()

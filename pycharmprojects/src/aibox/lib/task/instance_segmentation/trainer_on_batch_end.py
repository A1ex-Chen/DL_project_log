def on_batch_end(self, epoch: int, num_batches_in_epoch: int, n_batch: int,
    loss: float, anchor_objectness_loss: float, anchor_transformer_loss:
    float, proposal_class_loss: float, proposal_transformer_loss: float,
    mask_loss: float):
    global_batch = (epoch - 1) * num_batches_in_epoch + n_batch
    lr = self.scheduler.get_lr()[0]
    self.losses.append(loss)
    self.anchor_objectness_losses.append(anchor_objectness_loss)
    self.anchor_transformer_losses.append(anchor_transformer_loss)
    self.proposal_class_losses.append(proposal_class_loss)
    self.proposal_transformer_losses.append(proposal_transformer_loss)
    self.mask_losses.append(mask_loss)
    self.batches_counter += 1
    self.summary_writer.add_scalar('loss/loss', loss, global_batch)
    self.summary_writer.add_scalar('learning_rate', lr, global_batch)
    if (n_batch % self.config.num_batches_to_display == 0 or n_batch ==
        num_batches_in_epoch):
        elapsed_time = time.time() - self.time_checkpoint
        num_batches_per_sec = self.batches_counter / elapsed_time
        num_samples_per_sec = num_batches_per_sec * self.config.batch_size
        eta = ((self.config.num_epochs_to_finish - epoch) *
            num_batches_in_epoch + num_batches_in_epoch - n_batch
            ) / num_batches_per_sec / 3600
        avg_loss = sum(self.losses) / len(self.losses)
        avg_anchor_objectness_loss = sum(self.anchor_objectness_losses) / len(
            self.anchor_objectness_losses)
        avg_anchor_transformer_loss = sum(self.anchor_transformer_losses
            ) / len(self.anchor_transformer_losses)
        avg_proposal_class_loss = sum(self.proposal_class_losses) / len(self
            .proposal_class_losses)
        avg_proposal_transformer_loss = sum(self.proposal_transformer_losses
            ) / len(self.proposal_transformer_losses)
        avg_mask_loss = sum(self.mask_losses) / len(self.mask_losses)
        self.db.insert_log_table(DB.Log(global_batch, status=DB.Log.Status.
            RUNNING, datetime=int(time.time()), epoch=epoch, total_epoch=
            self.config.num_epochs_to_finish, batch=n_batch, total_batch=
            num_batches_in_epoch, avg_loss=avg_loss, learning_rate=lr,
            samples_per_sec=num_samples_per_sec, eta_hrs=eta))
        self.db.insert_instance_segmentation_log_table(DB.
            InstanceSegmentationLog(avg_anchor_objectness_loss=
            avg_anchor_objectness_loss, avg_anchor_transformer_loss=
            avg_anchor_transformer_loss, avg_proposal_class_loss=
            avg_proposal_class_loss, avg_proposal_transformer_loss=
            avg_proposal_transformer_loss, avg_mask_loss=avg_mask_loss))
        self.logger.i(
            f'[Epoch ({epoch}/{self.config.num_epochs_to_finish}) Batch ({n_batch}/{num_batches_in_epoch})] Avg. Loss = {avg_loss:.6f}, Learning Rate = {lr:.8f} ({num_samples_per_sec:.2f} samples/sec; ETA {eta:.2f} hrs)'
            )
        if self.profile_enabled:
            self.summary_writer.add_scalar('profile/num_samples_per_sec',
                num_samples_per_sec, global_batch)
            for i, handle in self.global_device_id_to_handle_dict.items():
                device_util_rates = nvidia_smi.nvmlDeviceGetUtilizationRates(
                    handle)
                self.summary_writer.add_scalar(f'profile/device_usage/{i}',
                    device_util_rates.gpu, global_batch)
                self.summary_writer.add_scalar(f'profile/device_memory/{i}',
                    device_util_rates.memory, global_batch)
        self.time_checkpoint = time.time()
        self.batches_counter = 0

def on_batch_end(self, epoch: int, num_batches_in_epoch: int, n_batch: int,
    loss: float):
    global_batch = (epoch - 1) * num_batches_in_epoch + n_batch
    lr = self.scheduler.get_lr()[0]
    self.losses.append(loss)
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
        self.db.insert_log_table(DB.Log(global_batch, status=DB.Log.Status.
            RUNNING, datetime=int(time.time()), epoch=epoch, total_epoch=
            self.config.num_epochs_to_finish, batch=n_batch, total_batch=
            num_batches_in_epoch, avg_loss=avg_loss, learning_rate=lr,
            samples_per_sec=num_samples_per_sec, eta_hrs=eta))
        self.logger.i(
            f'[Epoch ({epoch}/{self.config.num_epochs_to_finish}) Batch ({n_batch}/{num_batches_in_epoch})] Avg. Loss = {avg_loss:.6f}, Learning Rate = {lr:.8f} ({num_samples_per_sec:.2f} samples/sec; ETA {eta:.2f} hrs)'
            )
        self.time_checkpoint = time.time()
        self.batches_counter = 0

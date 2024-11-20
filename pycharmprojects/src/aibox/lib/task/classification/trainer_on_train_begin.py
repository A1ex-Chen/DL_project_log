def on_train_begin(self, num_batches_in_epoch: int):
    self.db.insert_log_table(DB.Log(global_batch=0, status=DB.Log.Status.
        INITIALIZED, datetime=int(time.time()), epoch=0, total_epoch=self.
        config.num_epochs_to_finish, batch=0, total_batch=
        num_batches_in_epoch, avg_loss=-1, learning_rate=-1,
        samples_per_sec=-1, eta_hrs=-1))

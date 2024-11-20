def insert_checkpoint_table(self, checkpoint: Checkpoint):
    cursor = self._connection.cursor()
    cursor.execute(DB.SQL_INSERT_CHECKPOINT_TABLE, (checkpoint.epoch,
        checkpoint.avg_loss, self.Checkpoint.serialize_metrics(checkpoint.
        metrics), checkpoint.is_best, checkpoint.is_available, checkpoint.
        task_name.value))
    self._connection.commit()

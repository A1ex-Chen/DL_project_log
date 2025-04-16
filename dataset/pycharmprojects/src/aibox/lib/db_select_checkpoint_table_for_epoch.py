def select_checkpoint_table_for_epoch(self, epoch: int) ->Checkpoint:
    cursor = self._connection.cursor()
    row = next(cursor.execute(DB.SQL_SELECT_CHECKPOINT_TABLE_FOR_EPOCH, (
        epoch,)))
    checkpoint = DB.Checkpoint(epoch=row[0], avg_loss=row[1], metrics=self.
        Checkpoint.deserialize_metrics(row[2]), is_best=row[3],
        is_available=row[4], task_name=Task.Name(row[5]))
    return checkpoint

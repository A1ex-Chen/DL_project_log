def select_checkpoint_table(self) ->List[Checkpoint]:
    cursor = self._connection.cursor()
    checkpoints = []
    for row in cursor.execute(DB.SQL_SELECT_CHECKPOINT_TABLE):
        checkpoints.append(DB.Checkpoint(epoch=row[0], avg_loss=row[1],
            metrics=self.Checkpoint.deserialize_metrics(row[2]), is_best=
            row[3], is_available=row[4], task_name=Task.Name(row[5])))
    return checkpoints

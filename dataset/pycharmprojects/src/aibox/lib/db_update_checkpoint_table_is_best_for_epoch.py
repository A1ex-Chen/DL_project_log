def update_checkpoint_table_is_best_for_epoch(self, is_best: bool, epoch: int):
    cursor = self._connection.cursor()
    cursor.execute(DB.SQL_UPDATE_CHECKPOINT_TABLE_IS_BEST_FOR_EPOCH, (
        is_best, epoch))
    self._connection.commit()

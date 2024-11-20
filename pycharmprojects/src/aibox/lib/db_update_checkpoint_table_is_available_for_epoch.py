def update_checkpoint_table_is_available_for_epoch(self, is_available: bool,
    epoch: int):
    cursor = self._connection.cursor()
    cursor.execute(DB.SQL_UPDATE_CHECKPOINT_TABLE_IS_AVAILABLE_FOR_EPOCH, (
        is_available, epoch))
    self._connection.commit()

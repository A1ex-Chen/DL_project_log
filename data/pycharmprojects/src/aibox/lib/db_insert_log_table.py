def insert_log_table(self, log: Log):
    cursor = self._connection.cursor()
    cursor.execute(DB.SQL_INSERT_LOG_TABLE, (log.global_batch, log.status.
        value, log.datetime, log.epoch, log.total_epoch, log.batch, log.
        total_batch, log.avg_loss, log.learning_rate, log.samples_per_sec,
        log.eta_hrs, log.exception if log.exception is None else DB.Log.
        serialize_exception(log.exception)))
    self._connection.commit()

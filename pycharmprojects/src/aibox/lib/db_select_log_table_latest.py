def select_log_table_latest(self) ->Log:
    cursor = self._connection.cursor()
    row = next(cursor.execute(DB.SQL_SELECT_LOG_TABLE_LATEST))
    log = DB.Log(global_batch=row[1], status=DB.Log.Status(row[2]),
        datetime=row[3], epoch=row[4], total_epoch=row[5], batch=row[6],
        total_batch=row[7], avg_loss=row[8], learning_rate=row[9],
        samples_per_sec=row[10], eta_hrs=row[11], exception=row[12] if row[
        12] is None else DB.Log.deserialize_exception(row[12]))
    return log

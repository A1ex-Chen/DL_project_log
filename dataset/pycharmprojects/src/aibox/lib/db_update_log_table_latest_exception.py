def update_log_table_latest_exception(self, exception: Log.Exception):
    cursor = self._connection.cursor()
    cursor.execute(DB.SQL_UPDATE_LOG_TABLE_LATEST_EXCEPTION, (int(time.time
        ()), DB.Log.Status.EXCEPTION.value, DB.Log.serialize_exception(
        exception)))
    self._connection.commit()

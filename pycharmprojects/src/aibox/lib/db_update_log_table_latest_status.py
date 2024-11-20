def update_log_table_latest_status(self, status: Log.Status):
    cursor = self._connection.cursor()
    cursor.execute(DB.SQL_UPDATE_LOG_TABLE_LATEST_STATUS, (int(time.time()),
        status.value))
    self._connection.commit()

def __init__(self, path_to_db: str):
    super().__init__()
    self._connection = sqlite3.connect(path_to_db)
    cursor = self._connection.cursor()
    cursor.execute(DB.SQL_CREATE_LOG_TABLE)
    cursor.execute(DB.SQL_CREATE_CHECKPOINT_TABLE)
    self._connection.commit()

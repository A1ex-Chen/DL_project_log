def __init__(self, path_to_db: str):
    super().__init__(path_to_db)
    cursor = self._connection.cursor()
    cursor.execute(DB.SQL_CREATE_INSTANCE_SEGMENTATION_LOG_TABLE)
    self._connection.commit()

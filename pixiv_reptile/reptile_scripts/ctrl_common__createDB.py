def _createDB(iPainterId):
    sDBPath = os.path.join(defines.PICTURES_PATH, defines.PICTURES_DB.
        format(iPainterId))
    if os.path.exists(sDBPath):
        return
    conn = sqlite3.connect(sDBPath)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS pictures (
            picture_id TEXT PRIMARY KEY
        )
    """
        )
    conn.commit()
    conn.close()

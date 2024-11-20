def CheckHavePicture(iPainterId, iPictureID):
    sDBPath = os.path.join(defines.PICTURES_PATH, defines.PICTURES_DB.
        format(iPainterId))
    if not os.path.exists(sDBPath):
        _createDB(iPainterId)
        return False
    conn = sqlite3.connect(sDBPath)
    cursor = conn.cursor()
    cursor.execute('SELECT 1 FROM pictures WHERE picture_id = ?', (str(
        iPictureID),))
    rows = cursor.fetchall()
    conn.close()
    return bool(len(rows) > 0)

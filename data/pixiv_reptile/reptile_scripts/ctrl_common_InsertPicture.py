def InsertPicture(iPainterId, iPictureID):
    sDBPath = os.path.join(defines.PICTURES_PATH, defines.PICTURES_DB.
        format(iPainterId))
    conn = sqlite3.connect(sDBPath)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO pictures (picture_id) VALUES (?)', (str(
        iPictureID),))
    conn.commit()
    conn.close()

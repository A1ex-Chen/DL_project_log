def __init__(self, dbFile):
    try:
        conn = sqlite3.connect(dbFile)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
    except:
        print('Error opening {}'.format(dbFile))
        sys.exit(1)
    self.conn = conn
    self.c = c

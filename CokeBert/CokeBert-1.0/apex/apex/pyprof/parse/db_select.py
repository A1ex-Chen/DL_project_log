def select(self, cmd):
    try:
        self.c.execute(cmd)
        rows = [dict(row) for row in self.c.fetchall()]
    except sqlite3.Error as e:
        print(e)
        sys.exit(1)
    except:
        print('Uncaught error in SQLite access while executing {}'.format(cmd))
        sys.exit(1)
    return rows

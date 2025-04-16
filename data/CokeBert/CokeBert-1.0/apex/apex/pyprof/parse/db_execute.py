def execute(self, cmd):
    try:
        self.c.execute(cmd)
    except sqlite3.Error as e:
        print(e)
        sys.exit(1)
    except:
        print('Uncaught error in SQLite access while executing {}'.format(cmd))
        sys.exit(1)

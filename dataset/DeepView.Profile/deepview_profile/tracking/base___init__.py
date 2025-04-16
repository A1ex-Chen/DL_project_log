def __init__(self, file=None):
    database_file = file if file is not None else ':memory:'
    self._connection = sqlite3.connect(database_file, check_same_thread=False)
    self._create_report_tables()

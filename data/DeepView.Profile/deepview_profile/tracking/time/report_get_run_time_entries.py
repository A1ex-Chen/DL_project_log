def get_run_time_entries(self, path_prefix=None):
    cursor = self._connection.cursor()
    return map(lambda row: RunTimeEntry(*row), cursor.execute(queries.
        get_run_time_entries_with_context))

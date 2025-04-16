def get_latest_n_entries_of_entry_point(self, n: int, entry_point: str) ->list:
    """
        Gets the n latest entries of a given entry point
        """
    params = [entry_point, n]
    cursor = self.database_connection.cursor()
    results = cursor.execute(
        'SELECT * FROM ENERGY WHERE entry_point=? ORDER BY ts DESC LIMIT ?;',
        params).fetchall()
    return results

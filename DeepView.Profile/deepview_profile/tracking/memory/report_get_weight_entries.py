def get_weight_entries(self, path_prefix=None):
    cursor = self._connection.cursor()
    return map(lambda row: WeightEntry(*row), cursor.execute(queries.
        get_weight_entries_with_context))

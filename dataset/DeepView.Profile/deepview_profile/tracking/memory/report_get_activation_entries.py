def get_activation_entries(self, path_prefix=None):
    cursor = self._connection.cursor()
    return map(lambda row: ActivationEntry(*row), cursor.execute(queries.
        get_activation_entries_with_context))

def write_line(self, values):
    """
        Write the values in the inner table.

        Args:
            values (:obj:`Dict[str, float]`): The values to display.
        """
    if self.inner_table is None:
        self.inner_table = [list(values.keys()), list(values.values())]
    else:
        columns = self.inner_table[0]
        if len(self.inner_table) == 1:
            for key in values.keys():
                if key not in columns:
                    columns.append(key)
            self.inner_table[0] = columns
        self.inner_table.append([values[c] for c in columns])

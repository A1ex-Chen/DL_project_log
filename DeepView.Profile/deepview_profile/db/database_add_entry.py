def add_entry(self, entry: list) ->bool:
    """
        Validates an entry and then adds that entry into the Energy table. Note that
        current timestamp is added by this function. Returns False if the entry is
        not a valid format, or if the insertion failed. Else returns True
        """
    if self.is_valid_entry(entry):
        try:
            entry.append(datetime.datetime.now())
            cursor = self.database_connection.cursor()
            cursor.execute('INSERT INTO ENERGY VALUES(?, ?, ?, ?, ?)', entry)
            self.database_connection.commit()
            return True
        except sqlite3.IntegrityError as e:
            print(e)
            return False
    else:
        return False

@staticmethod
def is_valid_entry(entry: list) ->bool:
    """
        Validates an entry in the Energy table by testing if the length is 3,
        and the types match the columns. Note that timestamp is not part of the entry.
        Returns True if it is valid, else False
        """
    return len(entry) == 4 and type(entry[0]) == str and type(entry[1]
        ) == float and type(entry[2]) == float and type(entry[3]) == int

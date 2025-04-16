@staticmethod
def is_valid_entry_with_timestamp(entry: list) ->bool:
    """
        Validates an entry in the Energy table by testing if the length is 4,
        and the types match the columns. Returns True if it is valid, else False
        """
    return len(entry) == 5 and type(entry[0]) == str and type(entry[1]
        ) == float and type(entry[2]) == float and type(entry[3]
        ) == int and type(entry[4]) == datetime.datetime

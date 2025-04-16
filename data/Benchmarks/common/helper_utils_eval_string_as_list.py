def eval_string_as_list(str_read, separator, dtype):
    """Parse a string and convert it into a list of lists.

    Parameters
    ----------
    str_read : string
        String read (from configuration file or command line, for example)
    separator : character
        Character that specifies the separation between the lists
    dtype : data type
        Data type to decode the elements of the list

    Return
    ----------
    decoded_list : list
        List extracted from string and with elements of the
        specified type.
    """
    ldtype = dtype
    if ldtype is None:
        ldtype = np.int32
    decoded_list = []
    out_list = str_read.split(separator)
    for el in out_list:
        decoded_list.append(ldtype(el))
    return decoded_list

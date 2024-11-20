def eval_string_as_list_of_lists(str_read, separator_out, separator_in, dtype):
    """Parse a string and convert it into a list of lists.

    Parameters
    ----------
    str_read : string
        String read (from configuration file or command line, for example)
    separator_out : character
        Character that specifies the separation between the outer level lists
    separator_in : character
        Character that specifies the separation between the inner level lists
    dtype : data type
        Data type to decode the elements of the lists

    Return
    ----------
    decoded_list : list
        List of lists extracted from string and with elements of the specified type.
    """
    ldtype = dtype
    if ldtype is None:
        ldtype = np.int32
    decoded_list = []
    out_list = str_read.split(separator_out)
    for line in out_list:
        in_list = []
        elem = line.split(separator_in)
        for el in elem:
            in_list.append(ldtype(el))
        decoded_list.append(in_list)
    return decoded_list

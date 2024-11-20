def __init__(self, option_strings, dest, type, **kwargs):
    """Initialize a ListOfListsAction object. If no type is specified,
        an integer is assumed by default as the type for the elements
        of the list-of-lists.

        Parameters
        ----------
        option_strings : string
            String to parse
        dest : object
            Object to store the output (in this case the parsed list-of-lists).
        type : data type
            Data type to decode the elements of the lists.
            Defaults to np.int32.
        kwargs : object
            Python object containing other argparse.Action parameters.

        """
    super(ListOfListsAction, self).__init__(option_strings, dest, **kwargs)
    self.dtype = type
    if self.dtype is None:
        self.dtype = np.int32

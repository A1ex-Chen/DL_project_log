def __call__(self, parser, namespace, values, option_string=None):
    """This function overrides the __call__ method of the base
        argparse.Action class.

        This function implements the action of the ListOfListAction
        class by parsing an input string (command-line option or argument)
        and maping it into a list-of-lists. The resulting list-of-lists is
        added to the namespace of parsed arguments. The parsing assumes that
        the separator between lists is a colon ':' and the separator inside
        the list is a comma ','. The values of the list are casted to the
        type specified at the object initialization.

        Parameters
        ----------
        parser : ArgumentParser object
            Object that contains this action
        namespace : Namespace object
            Namespace object that will be returned by the parse_args()
            function.
        values : string
            The associated command-line arguments converted to string type
            (i.e. input).
        option_string : string
            The option string that was used to invoke this action. (optional)

        """
    decoded_list = []
    removed1 = values.replace('[', '')
    removed2 = removed1.replace(']', '')
    out_list = removed2.split(':')
    for line in out_list:
        in_list = []
        elem = line.split(',')
        for el in elem:
            in_list.append(self.dtype(el))
        decoded_list.append(in_list)
    setattr(namespace, self.dest, decoded_list)

def str2bool(v):
    """This is taken from:
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    Because type=bool is not interpreted as a bool and action='store_true' cannot be
    undone.

    Parameters
    ----------
    v : string
        String to interpret

    Return
    ----------
    Boolean value. It raises and exception if the provided string cannot
    be interpreted as a boolean type.
    Strings recognized as boolean True :
        'yes', 'true', 't', 'y', '1' and uppercase versions (where applicable).
    Strings recognized as boolean False :
        'no', 'false', 'f', 'n', '0' and uppercase versions (where applicable).
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

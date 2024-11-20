def parse_common(parser):
    """Functionality to parse options.
    Parameters
    ----------
    parser : ArgumentParser object
        Current parser
    """
    for lst in registered_conf:
        parser = parse_from_dictlist(lst, parser)
    return parser

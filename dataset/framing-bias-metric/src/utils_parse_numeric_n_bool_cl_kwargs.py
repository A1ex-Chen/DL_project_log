def parse_numeric_n_bool_cl_kwargs(unparsed_args: List[str]) ->Dict[str,
    Union[int, float, bool]]:
    """
    Parse an argv list of unspecified command line args to a dict.
    Assumes all values are either numeric or boolean in the form of true/false.
    """
    result = {}
    assert len(unparsed_args
        ) % 2 == 0, f'got odd number of unparsed args: {unparsed_args}'
    num_pairs = len(unparsed_args) // 2
    for pair_num in range(num_pairs):
        i = 2 * pair_num
        assert unparsed_args[i].startswith('--')
        if unparsed_args[i + 1].lower() == 'true':
            value = True
        elif unparsed_args[i + 1].lower() == 'false':
            value = False
        else:
            try:
                value = int(unparsed_args[i + 1])
            except ValueError:
                value = float(unparsed_args[i + 1])
        result[unparsed_args[i][2:]] = value
    return result

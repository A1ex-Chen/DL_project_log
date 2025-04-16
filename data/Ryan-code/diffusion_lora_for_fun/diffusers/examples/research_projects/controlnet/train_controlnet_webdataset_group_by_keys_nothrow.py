def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=
    None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample['fname'], filesample['data']
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        if current_sample is None or prefix != current_sample['__key__'
            ] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = {'__key__': prefix, '__url__': filesample[
                '__url__']}
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample

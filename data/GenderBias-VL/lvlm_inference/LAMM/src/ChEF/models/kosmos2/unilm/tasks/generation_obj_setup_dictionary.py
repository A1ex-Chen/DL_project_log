@classmethod
def setup_dictionary(cls, args, **kwargs):
    dictionary = None
    output_dictionary = None
    paths = utils.split_paths(args.data)
    assert len(paths) > 0
    if len(args.dict_path) > 0:
        dictionary = Dictionary.load(args.dict_path)
    else:
        dictionary = Dictionary.load(os.path.join(paths[0], 'dict.txt'))
    dictionary.add_symbol('<mask>')
    for special_symbol in add_location_symbols(args.location_bin_size, args
        .locate_special_token):
        dictionary.add_symbol(special_symbol)
    dictionary.pad_to_multiple_(args.required_batch_size_multiple)
    output_dictionary = dictionary
    logger.info('dictionary from {}: {} types'.format(args.dict_path, len(
        dictionary)))
    return dictionary, output_dictionary

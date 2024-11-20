def load_from_cache(self, cache, params):
    """NOTE: How does this function return an error? (False?) -Wozniak"""
    param_fname = '{}.params.json'.format(cache)
    if not os.path.isfile(param_fname):
        logger.warning('Cache parameter file does not exist: %s', param_fname)
        return False
    with open(param_fname) as param_file:
        try:
            cached_params = json.load(param_file)
        except json.JSONDecodeError as e:
            logger.warning('Could not decode parameter file %s: %s',
                param_fname, e)
            return False
    ignore_keys = ['cache', 'partition_by', 'single', 'use_exported_data']
    equal, diffs = dict_compare(params, cached_params, ignore_keys)
    if not equal:
        logger.warning(
            'Cache parameter mismatch: %s\nSaved: %s\nAttempted to load: %s',
            diffs, cached_params, params)
        logger.warning('\nRemove %s to rebuild data cache.\n', param_fname)
        raise ValueError('Could not load from a cache with incompatible keys:',
            diffs)
    else:
        fname = '{}.pkl'.format(cache)
        if not os.path.isfile(fname):
            logger.warning('Cache file does not exist: %s', fname)
            return False
        with open(fname, 'rb') as f:
            obj = pickle.load(f)
        self.__dict__.update(obj.__dict__)
        logger.info('Loaded data from cache: %s', fname)
        return True
    return False

def save_to_cache(self, cache, params):
    for k in ['self', 'cache', 'single']:
        if k in params:
            del params[k]
    dirname = os.path.dirname(cache)
    if dirname and not os.path.exists(dirname):
        logger.debug('Creating directory for cache: %s', dirname)
        os.makedirs(dirname)
    param_fname = '{}.params.json'.format(cache)
    with open(param_fname, 'w') as param_file:
        json.dump(params, param_file, sort_keys=True)
    fname = '{}.pkl'.format(cache)
    with open(fname, 'wb') as f:
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    logger.info('Saved data to cache: %s', fname)

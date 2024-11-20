def get_file(url):
    fname = os.path.basename(url)
    return candle.get_file(fname, origin=url, cache_subdir='Examples')

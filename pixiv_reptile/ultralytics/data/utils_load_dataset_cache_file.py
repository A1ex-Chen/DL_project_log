def load_dataset_cache_file(path):
    """Load an Ultralytics *.cache dictionary from path."""
    import gc
    gc.disable()
    cache = np.load(str(path), allow_pickle=True).item()
    gc.enable()
    return cache

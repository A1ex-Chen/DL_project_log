def hash_key(self, model_key: str, item: Any, kwargs: Dict[str, Any]):
    pickled = pickle.dumps((item, kwargs))
    return cachetools.keys.hashkey((model_key, pickled))

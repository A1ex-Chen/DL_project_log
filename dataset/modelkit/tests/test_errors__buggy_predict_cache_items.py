def _buggy_predict_cache_items(*args, **kwargs):
    raise CustomError
    yield None

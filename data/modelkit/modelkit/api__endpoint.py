def _endpoint(item: List[item_type]=fastapi.Body(...), model=fastapi.
    Depends(lambda : self.lib.get(model.configuration_key))):
    return model.predict_batch(item)

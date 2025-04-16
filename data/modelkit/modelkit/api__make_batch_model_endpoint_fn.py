def _make_batch_model_endpoint_fn(self, model, item_type):
    if isinstance(model, AsyncModel):

        async def _aendpoint(item: List[item_type]=fastapi.Body(...), model
            =fastapi.Depends(lambda : self.lib.get(model.configuration_key))):
            return await model.predict_batch(item)
        return _aendpoint

    def _endpoint(item: List[item_type]=fastapi.Body(...), model=fastapi.
        Depends(lambda : self.lib.get(model.configuration_key))):
        return model.predict_batch(item)
    return _endpoint

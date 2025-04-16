def _make_model_endpoint_fn(self, model, item_type):
    if isinstance(model, AsyncModel):

        async def _aendpoint(item: item_type=fastapi.Body(...), model=
            fastapi.Depends(lambda : self.lib.get(model.configuration_key))):
            return await model.predict(item)
        return _aendpoint

    def _endpoint(item: item_type=fastapi.Body(...), model=fastapi.Depends(
        lambda : self.lib.get(model.configuration_key))):
        return model.predict(item)
    return _endpoint

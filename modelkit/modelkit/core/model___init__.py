def __init__(self, async_model: AsyncModel[ItemType, ReturnType]):
    self.async_model = async_model
    self.predict = AsyncToSync(self.async_model.predict)
    self.predict_batch = AsyncToSync(self.async_model.predict_batch)
    self._loaded: bool = True

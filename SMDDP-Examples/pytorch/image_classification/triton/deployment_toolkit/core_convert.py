@abc.abstractmethod
def convert(self, model: Model, dataloader_fn) ->Model:
    raise NotImplementedError()

@property
def tensor_inputs(self) ->List[str]:
    return [input for callback in self.callbacks for input in callback.
        tensor_inputs]

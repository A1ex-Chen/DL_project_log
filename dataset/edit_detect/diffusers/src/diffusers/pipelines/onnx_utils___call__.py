def __call__(self, **kwargs):
    inputs = {k: np.array(v) for k, v in kwargs.items()}
    return self.model.run(None, inputs)

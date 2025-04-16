def __init__(self, model_path: str='', input_vector_size: (int | None)=None,
    output_vector_size: (int | None)=None) ->None:
    super().__init__(input_vector_size, output_vector_size)
    self.model_path = model_path
    self.load(model_path)

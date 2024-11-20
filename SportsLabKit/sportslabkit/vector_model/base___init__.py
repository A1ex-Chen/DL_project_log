def __init__(self, input_vector_size: (int | None)=None, output_vector_size:
    (int | None)=None) ->None:
    """Initialize the BaseVectorModel.

        Args:
            input_vector_size (Optional[int]): The size of the input vector. None to bypass validation.
            output_vector_size (Optional[int]): The size of the output vector. None to bypass validation.
        """
    super().__init__()
    self.input_vector_size = input_vector_size
    self.output_vector_size = output_vector_size
    self.model = None

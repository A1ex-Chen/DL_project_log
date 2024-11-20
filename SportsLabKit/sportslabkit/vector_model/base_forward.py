@abstractmethod
def forward(self, inputs: Vector, **kwargs: Any) ->Vector:
    """Define the forward pass of the model. Must be overridden by subclasses.

        Args:
            inputs (Vector): The input data.
            **kwargs (Any): Additional arguments to be passed to the forward method.

        Returns:
            Vector: The output data.
        """
    raise NotImplementedError(
        'The forward method must be implemented by subclasses.')

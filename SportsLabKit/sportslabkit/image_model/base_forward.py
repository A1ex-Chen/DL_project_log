@abstractmethod
def forward(self, x: np.ndarray):
    """
        Forward must be overridden by subclasses. The overriding method should define the forward pass of the model. The model will receive a 4-dimensional numpy array representing the images. As for the output, the model is expected to return something that can be converted into a 2-dimensional numpy array.

        Args:
            x (np.ndarray): input image
        """
    raise NotImplementedError

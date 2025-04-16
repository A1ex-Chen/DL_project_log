@abstractmethod
def forward(self, x):
    """
        Args:
            x (Tensor): input tensor
        Returns:
            Tensor: output tensor
        """
    raise NotImplementedError

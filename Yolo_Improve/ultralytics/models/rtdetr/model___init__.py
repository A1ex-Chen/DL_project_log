def __init__(self, model='rtdetr-l.pt') ->None:
    """
        Initializes the RT-DETR model with the given pre-trained model file. Supports .pt and .yaml formats.

        Args:
            model (str): Path to the pre-trained model. Defaults to 'rtdetr-l.pt'.

        Raises:
            NotImplementedError: If the model file extension is not 'pt', 'yaml', or 'yml'.
        """
    super().__init__(model=model, task='detect')

def __init__(self, model='sam_b.pt') ->None:
    """
        Initializes the SAM model with a pre-trained model file.

        Args:
            model (str): Path to the pre-trained SAM model file. File should have a .pt or .pth extension.

        Raises:
            NotImplementedError: If the model file extension is not .pt or .pth.
        """
    if model and Path(model).suffix not in {'.pt', '.pth'}:
        raise NotImplementedError(
            'SAM prediction requires pre-trained *.pt or *.pth model.')
    super().__init__(model=model, task='segment')

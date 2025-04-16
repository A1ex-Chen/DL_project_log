def __init__(self, device='cuda'):
    """
        Parameters
        ----------
        device : str, optional
            The device to run the model on ('cpu' or 'cuda'). By default, 'cuda' is used.
        """
    self.device = device
    self.model = BitNetTransformer(num_tokens=256, dim=512, depth=8)
    self.model = AutoregressiveWrapper(self.model, max_seq_len=1024)
    self.model.to(self.device)

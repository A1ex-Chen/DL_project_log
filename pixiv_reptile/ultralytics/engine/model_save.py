def save(self, filename: Union[str, Path]='saved_model.pt', use_dill=True
    ) ->None:
    """
        Saves the current model state to a file.

        This method exports the model's checkpoint (ckpt) to the specified filename.

        Args:
            filename (str | Path): The name of the file to save the model to. Defaults to 'saved_model.pt'.
            use_dill (bool): Whether to try using dill for serialization if available. Defaults to True.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
    self._check_is_pytorch_model()
    from datetime import datetime
    from ultralytics import __version__
    updates = {'date': datetime.now().isoformat(), 'version': __version__,
        'license': 'AGPL-3.0 License (https://ultralytics.com/license)',
        'docs': 'https://docs.ultralytics.com'}
    torch.save({**self.ckpt, **updates}, filename, use_dill=use_dill)

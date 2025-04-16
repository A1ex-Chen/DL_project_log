def embed(self, source: Union[str, Path, int, list, tuple, np.ndarray,
    torch.Tensor]=None, stream: bool=False, **kwargs) ->list:
    """
        Generates image embeddings based on the provided source.

        This method is a wrapper around the 'predict()' method, focusing on generating embeddings from an image source.
        It allows customization of the embedding process through various keyword arguments.

        Args:
            source (str | int | PIL.Image | np.ndarray): The source of the image for generating embeddings.
                The source can be a file path, URL, PIL image, numpy array, etc. Defaults to None.
            stream (bool): If True, predictions are streamed. Defaults to False.
            **kwargs (any): Additional keyword arguments for configuring the embedding process.

        Returns:
            (List[torch.Tensor]): A list containing the image embeddings.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
    if not kwargs.get('embed'):
        kwargs['embed'] = [len(self.model.model) - 2]
    return self.predict(source, stream, **kwargs)

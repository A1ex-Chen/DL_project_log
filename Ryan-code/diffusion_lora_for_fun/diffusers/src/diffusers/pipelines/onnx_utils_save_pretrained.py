def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
    """
        Save a model to a directory, so that it can be re-loaded using the [`~OnnxModel.from_pretrained`] class
        method.:

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
        """
    if os.path.isfile(save_directory):
        logger.error(
            f'Provided path ({save_directory}) should be a directory, not a file'
            )
        return
    os.makedirs(save_directory, exist_ok=True)
    self._save_pretrained(save_directory, **kwargs)

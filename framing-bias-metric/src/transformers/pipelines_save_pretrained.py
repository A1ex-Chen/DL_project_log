def save_pretrained(self, save_directory: str):
    """
        Save the pipeline's model and tokenizer.

        Args:
            save_directory (:obj:`str`):
                A path to the directory where to saved. It will be created if it doesn't exist.
        """
    if os.path.isfile(save_directory):
        logger.error('Provided path ({}) should be a directory, not a file'
            .format(save_directory))
        return
    os.makedirs(save_directory, exist_ok=True)
    self.model.save_pretrained(save_directory)
    self.tokenizer.save_pretrained(save_directory)
    if self.modelcard is not None:
        self.modelcard.save_pretrained(save_directory)

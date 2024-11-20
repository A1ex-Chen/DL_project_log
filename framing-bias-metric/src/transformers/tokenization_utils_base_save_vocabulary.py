def save_vocabulary(self, save_directory: str, filename_prefix: Optional[
    str]=None) ->Tuple[str]:
    """
        Save only the vocabulary of the tokenizer (vocabulary + added tokens).

        This method won't save the configuration and special token mappings of the tokenizer. Use
        :meth:`~transformers.PreTrainedTokenizerFast._save_pretrained` to save the whole state of the tokenizer.

        Args:
            save_directory (:obj:`str`):
                The directory in which to save the vocabulary.
            filename_prefix (:obj:`str`, `optional`):
                An optional prefix to add to the named of the saved files.

        Returns:
            :obj:`Tuple(str)`: Paths to the files saved.
        """
    raise NotImplementedError

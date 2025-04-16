def prepare_for_tokenization(self, text: str, is_split_into_words: bool=
    False, **kwargs) ->Tuple[str, Dict[str, Any]]:
    """
        Performs any necessary transformations before tokenization.

        This method should pop the arguments from kwargs and return the remaining :obj:`kwargs` as well. We test the
        :obj:`kwargs` at the end of the encoding process to be sure all the arguments have been used.

        Args:
            text (:obj:`str`):
                The text to prepare.
            is_split_into_words (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the text has been pretokenized.
            kwargs:
                Keyword arguments to use for the tokenization.

        Returns:
            :obj:`Tuple[str, Dict[str, Any]]`: The prepared text and the unused kwargs.
        """
    return text, kwargs

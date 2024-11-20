@property
def encodings(self) ->Optional[List[EncodingFast]]:
    """
        :obj:`Optional[List[tokenizers.Encoding]]`: The list all encodings from the tokenization process. Returns
        :obj:`None` if the input was tokenized through Python (i.e., not a fast) tokenizer.
        """
    return self._encodings

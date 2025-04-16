def __getitem__(self, item: Union[int, str]) ->Union[Any, EncodingFast]:
    """
        If the key is a string, returns the value of the dict associated to :obj:`key` ('input_ids', 'attention_mask',
        etc.).

        If the key is an integer, get the :obj:`tokenizers.Encoding` for batch item with index :obj:`key`.
        """
    if isinstance(item, str):
        return self.data[item]
    elif self._encodings is not None:
        return self._encodings[item]
    else:
        raise KeyError(
            'Indexing with integers (to access backend Encoding for a given batch index) is not available when using Python based tokenizers'
            )

@add_end_docstrings(ENCODE_KWARGS_DOCSTRING,
    """
            **kwargs: Passed along to the `.tokenize()` method.
        """
    ,
    """
        Returns:
            :obj:`List[int]`, :obj:`torch.Tensor`, :obj:`tf.Tensor` or :obj:`np.ndarray`: The tokenized ids of the
            text.
        """
    )
def encode(self, text: Union[TextInput, PreTokenizedInput, EncodedInput],
    text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]]=
    None, add_special_tokens: bool=True, padding: Union[bool, str,
    PaddingStrategy]=False, truncation: Union[bool, str, TruncationStrategy
    ]=False, max_length: Optional[int]=None, stride: int=0, return_tensors:
    Optional[Union[str, TensorType]]=None, **kwargs) ->List[int]:
    """
        Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

        Same as doing ``self.convert_tokens_to_ids(self.tokenize(text))``.

        Args:
            text (:obj:`str`, :obj:`List[str]` or :obj:`List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                ``tokenize`` method) or a list of integers (tokenized string ids using the ``convert_tokens_to_ids``
                method).
            text_pair (:obj:`str`, :obj:`List[str]` or :obj:`List[int]`, `optional`):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the ``tokenize`` method) or a list of integers (tokenized string ids using the
                ``convert_tokens_to_ids`` method).
        """
    encoded_inputs = self.encode_plus(text, text_pair=text_pair,
        add_special_tokens=add_special_tokens, padding=padding, truncation=
        truncation, max_length=max_length, stride=stride, return_tensors=
        return_tensors, **kwargs)
    return encoded_inputs['input_ids']

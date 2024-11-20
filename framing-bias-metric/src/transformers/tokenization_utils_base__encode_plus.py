def _encode_plus(self, text: Union[TextInput, PreTokenizedInput,
    EncodedInput], text_pair: Optional[Union[TextInput, PreTokenizedInput,
    EncodedInput]]=None, add_special_tokens: bool=True, padding_strategy:
    PaddingStrategy=PaddingStrategy.DO_NOT_PAD, truncation_strategy:
    TruncationStrategy=TruncationStrategy.DO_NOT_TRUNCATE, max_length:
    Optional[int]=None, stride: int=0, is_split_into_words: bool=False,
    pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[
    str, TensorType]]=None, return_token_type_ids: Optional[bool]=None,
    return_attention_mask: Optional[bool]=None, return_overflowing_tokens:
    bool=False, return_special_tokens_mask: bool=False,
    return_offsets_mapping: bool=False, return_length: bool=False, verbose:
    bool=True, **kwargs) ->BatchEncoding:
    raise NotImplementedError

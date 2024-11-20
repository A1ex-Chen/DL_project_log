@add_end_docstrings(ENCODE_KWARGS_DOCSTRING,
    ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
def __call__(self, text: Union[TextInput, PreTokenizedInput, List[TextInput
    ], List[PreTokenizedInput]], text_pair: Optional[Union[TextInput,
    PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]]=None,
    add_special_tokens: bool=True, padding: Union[bool, str,
    PaddingStrategy]=False, truncation: Union[bool, str, TruncationStrategy
    ]=False, max_length: Optional[int]=None, stride: int=0,
    is_split_into_words: bool=False, pad_to_multiple_of: Optional[int]=None,
    return_tensors: Optional[Union[str, TensorType]]=None,
    return_token_type_ids: Optional[bool]=None, return_attention_mask:
    Optional[bool]=None, return_overflowing_tokens: bool=False,
    return_special_tokens_mask: bool=False, return_offsets_mapping: bool=
    False, return_length: bool=False, verbose: bool=True, **kwargs
    ) ->BatchEncoding:
    """
        Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
        sequences.

        Args:
            text (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                :obj:`is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            text_pair (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                :obj:`is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
        """
    assert isinstance(text, str) or isinstance(text, (list, tuple)) and (
        len(text) == 0 or (isinstance(text[0], str) or isinstance(text[0],
        (list, tuple)) and (len(text[0]) == 0 or isinstance(text[0][0], str)))
        ), 'text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) or `List[List[str]]` (batch of pretokenized examples).'
    assert text_pair is None or isinstance(text_pair, str) or isinstance(
        text_pair, (list, tuple)) and (len(text_pair) == 0 or (isinstance(
        text_pair[0], str) or isinstance(text_pair[0], (list, tuple)) and (
        len(text_pair[0]) == 0 or isinstance(text_pair[0][0], str)))
        ), 'text_pair input must of type `str` (single example), `List[str]` (batch or single pretokenized example) or `List[List[str]]` (batch of pretokenized examples).'
    is_batched = bool(not is_split_into_words and isinstance(text, (list,
        tuple)) or is_split_into_words and isinstance(text, (list, tuple)) and
        text and isinstance(text[0], (list, tuple)))
    if is_batched:
        batch_text_or_text_pairs = list(zip(text, text_pair)
            ) if text_pair is not None else text
        return self.batch_encode_plus(batch_text_or_text_pairs=
            batch_text_or_text_pairs, add_special_tokens=add_special_tokens,
            padding=padding, truncation=truncation, max_length=max_length,
            stride=stride, is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of, return_tensors=
            return_tensors, return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping, return_length=
            return_length, verbose=verbose, **kwargs)
    else:
        return self.encode_plus(text=text, text_pair=text_pair,
            add_special_tokens=add_special_tokens, padding=padding,
            truncation=truncation, max_length=max_length, stride=stride,
            is_split_into_words=is_split_into_words, pad_to_multiple_of=
            pad_to_multiple_of, return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping, return_length=
            return_length, verbose=verbose, **kwargs)

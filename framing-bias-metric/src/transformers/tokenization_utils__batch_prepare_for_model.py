@add_end_docstrings(ENCODE_KWARGS_DOCSTRING,
    ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
def _batch_prepare_for_model(self, batch_ids_pairs: List[Union[
    PreTokenizedInputPair, Tuple[List[int], None]]], add_special_tokens:
    bool=True, padding_strategy: PaddingStrategy=PaddingStrategy.DO_NOT_PAD,
    truncation_strategy: TruncationStrategy=TruncationStrategy.
    DO_NOT_TRUNCATE, max_length: Optional[int]=None, stride: int=0,
    pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[str]=
    None, return_token_type_ids: Optional[bool]=None, return_attention_mask:
    Optional[bool]=None, return_overflowing_tokens: bool=False,
    return_special_tokens_mask: bool=False, return_length: bool=False,
    verbose: bool=True) ->BatchEncoding:
    """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens

        Args:
            batch_ids_pairs: list of tokenized input ids or input ids pairs
        """
    batch_outputs = {}
    for first_ids, second_ids in batch_ids_pairs:
        outputs = self.prepare_for_model(first_ids, second_ids,
            add_special_tokens=add_special_tokens, padding=PaddingStrategy.
            DO_NOT_PAD.value, truncation=truncation_strategy.value,
            max_length=max_length, stride=stride, pad_to_multiple_of=None,
            return_attention_mask=False, return_token_type_ids=
            return_token_type_ids, return_overflowing_tokens=
            return_overflowing_tokens, return_special_tokens_mask=
            return_special_tokens_mask, return_length=return_length,
            return_tensors=None, prepend_batch_axis=False, verbose=verbose)
        for key, value in outputs.items():
            if key not in batch_outputs:
                batch_outputs[key] = []
            batch_outputs[key].append(value)
    batch_outputs = self.pad(batch_outputs, padding=padding_strategy.value,
        max_length=max_length, pad_to_multiple_of=pad_to_multiple_of,
        return_attention_mask=return_attention_mask)
    batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)
    return batch_outputs

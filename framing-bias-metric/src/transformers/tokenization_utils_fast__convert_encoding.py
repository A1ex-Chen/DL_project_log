def _convert_encoding(self, encoding: EncodingFast, return_token_type_ids:
    Optional[bool]=None, return_attention_mask: Optional[bool]=None,
    return_overflowing_tokens: bool=False, return_special_tokens_mask: bool
    =False, return_offsets_mapping: bool=False, return_length: bool=False,
    verbose: bool=True) ->Tuple[Dict[str, Any], List[EncodingFast]]:
    """
        Convert the encoding representation (from low-level HuggingFace tokenizer output) to a python Dict and a list
        of encodings, take care of building a batch from overflowing tokens.

        Overflowing tokens are converted to additional examples (like batches) so the output values of the dict are
        lists (overflows) of lists (tokens).

        Output shape: (overflows, sequence length)
        """
    if return_token_type_ids is None:
        return_token_type_ids = 'token_type_ids' in self.model_input_names
    if return_attention_mask is None:
        return_attention_mask = 'attention_mask' in self.model_input_names
    if return_overflowing_tokens and encoding.overflowing is not None:
        encodings = [encoding] + encoding.overflowing
    else:
        encodings = [encoding]
    encoding_dict = defaultdict(list)
    for e in encodings:
        encoding_dict['input_ids'].append(e.ids)
        if return_token_type_ids:
            encoding_dict['token_type_ids'].append(e.type_ids)
        if return_attention_mask:
            encoding_dict['attention_mask'].append(e.attention_mask)
        if return_special_tokens_mask:
            encoding_dict['special_tokens_mask'].append(e.special_tokens_mask)
        if return_offsets_mapping:
            encoding_dict['offset_mapping'].append(e.offsets)
        if return_length:
            encoding_dict['length'].append(len(e.ids))
    return encoding_dict, encodings

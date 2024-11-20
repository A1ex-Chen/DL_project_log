def _pad(self, encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding
    ], max_length: Optional[int]=None, padding_strategy: PaddingStrategy=
    PaddingStrategy.DO_NOT_PAD, pad_to_multiple_of: Optional[int]=None,
    return_attention_mask: Optional[bool]=None) ->dict:
    """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs: Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                >= 7.5 (Volta).
            return_attention_mask: (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
    if return_attention_mask is None:
        return_attention_mask = 'attention_mask' in self.model_input_names
    if padding_strategy == PaddingStrategy.LONGEST:
        max_length = len(encoded_inputs['input_ids'])
    if (max_length is not None and pad_to_multiple_of is not None and 
        max_length % pad_to_multiple_of != 0):
        max_length = (max_length // pad_to_multiple_of + 1
            ) * pad_to_multiple_of
    needs_to_be_padded = (padding_strategy != PaddingStrategy.DO_NOT_PAD and
        len(encoded_inputs['input_ids']) != max_length)
    if needs_to_be_padded:
        difference = max_length - len(encoded_inputs['input_ids'])
        if self.padding_side == 'right':
            if return_attention_mask:
                encoded_inputs['attention_mask'] = [1] * len(encoded_inputs
                    ['input_ids']) + [0] * difference
            if 'token_type_ids' in encoded_inputs:
                encoded_inputs['token_type_ids'] = encoded_inputs[
                    'token_type_ids'] + [self.pad_token_type_id] * difference
            if 'special_tokens_mask' in encoded_inputs:
                encoded_inputs['special_tokens_mask'] = encoded_inputs[
                    'special_tokens_mask'] + [1] * difference
            encoded_inputs['input_ids'] = encoded_inputs['input_ids'] + [self
                .pad_token_id] * difference
        elif self.padding_side == 'left':
            if return_attention_mask:
                encoded_inputs['attention_mask'] = [0] * difference + [1
                    ] * len(encoded_inputs['input_ids'])
            if 'token_type_ids' in encoded_inputs:
                encoded_inputs['token_type_ids'] = [self.pad_token_type_id
                    ] * difference + encoded_inputs['token_type_ids']
            if 'special_tokens_mask' in encoded_inputs:
                encoded_inputs['special_tokens_mask'] = [1
                    ] * difference + encoded_inputs['special_tokens_mask']
            encoded_inputs['input_ids'] = [self.pad_token_id
                ] * difference + encoded_inputs['input_ids']
        else:
            raise ValueError('Invalid padding strategy:' + str(self.
                padding_side))
    elif return_attention_mask:
        encoded_inputs['attention_mask'] = [1] * len(encoded_inputs[
            'input_ids'])
    return encoded_inputs

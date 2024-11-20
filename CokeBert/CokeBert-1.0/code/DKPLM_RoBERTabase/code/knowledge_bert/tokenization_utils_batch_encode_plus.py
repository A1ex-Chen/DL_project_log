def batch_encode_plus(self, batch_text_or_text_pairs=None,
    add_special_tokens=False, max_length=None, stride=0,
    truncation_strategy='longest_first', return_tensors=None,
    return_input_lengths=False, return_attention_masks=False, **kwargs):
    """
        Returns a dictionary containing the encoded sequence or sequence pair and additional information:
        the mask for sequence classification and the overflowing elements if a ``max_length`` is specified.

        Args:
            batch_text_or_text_pairs: Batch of sequences or pair of sequences to be encoded.
                This can be a list of string/string-sequences/int-sequences or a list of pair of
                string/string-sequences/int-sequence (see details in encode_plus)
            add_special_tokens: if set to ``True``, the sequences will be encoded with the special tokens relative
                to their model.
            max_length: if set to a number, will limit the total sequence returned so that it has a maximum length.
                If there are overflowing tokens, those will be added to the returned dictionary`
            stride: if set to a number along with max_length, the overflowing tokens returned will contain some tokens
                from the main sequence returned. The value of this argument defines the number of additional tokens.
            truncation_strategy: string selected in the following options:
                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
                    starting from the longest one at each token (when there is a pair of input sequences)
                - 'only_first': Only truncate the first sequence
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_length)
            return_tensors: (optional) can be set to 'tf' or 'pt' to return respectively TensorFlow tf.constant
                or PyTorch torch.Tensor instead of a list of python integers.
            **kwargs: passed to the `self.tokenize()` method
        """
    batch_outputs = {}
    for ids_or_pair_ids in batch_text_or_text_pairs:
        if isinstance(ids_or_pair_ids, (list, tuple)):
            assert len(ids_or_pair_ids) == 2
            ids, pair_ids = ids_or_pair_ids
        else:
            ids, pair_ids = ids_or_pair_ids, None
        outputs = self.encode_plus(ids, pair_ids, add_special_tokens=
            add_special_tokens, max_length=max_length, stride=stride,
            truncation_strategy=truncation_strategy, return_tensors=None)
        if return_input_lengths:
            outputs['input_len'] = len(outputs['input_ids'])
        for key, value in outputs.items():
            if key not in batch_outputs:
                batch_outputs[key] = []
            batch_outputs[key].append(value)
    max_seq_len = max(map(len, batch_outputs['input_ids']))
    if return_attention_masks:
        batch_outputs['attention_mask'] = [([0] * len(v)) for v in
            batch_outputs['input_ids']]
    if return_tensors is not None:
        for key, value in batch_outputs.items():
            padded_value = value
            if key != 'input_len':
                padded_value = [(v + [self.pad_token_id if key ==
                    'input_ids' else 1] * (max_seq_len - len(v))) for v in
                    padded_value]
            if return_tensors == 'tf' and is_tf_available():
                batch_outputs[key] = tf.constant(padded_value)
            elif return_tensors == 'pt' and is_torch_available():
                batch_outputs[key] = torch.tensor(padded_value)
            elif return_tensors is not None:
                logger.warning(
                    'Unable to convert output to tensors format {}, PyTorch or TensorFlow is not available.'
                    .format(return_tensors))
    if return_attention_masks:
        if is_tf_available():
            batch_outputs['attention_mask'] = tf.abs(batch_outputs[
                'attention_mask'] - 1)
        else:
            batch_outputs['attention_mask'] = torch.abs(batch_outputs[
                'attention_mask'] - 1)
    return batch_outputs

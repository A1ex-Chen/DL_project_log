def mask_tokens(self, inputs: torch.Tensor) ->Tuple[torch.Tensor, torch.
    Tensor, torch.Tensor, torch.Tensor]:
    """
        The masked tokens to be predicted for a particular sequence are determined by the following algorithm:

            0. Start from the beginning of the sequence by setting ``cur_len = 0`` (number of tokens processed so far).
            1. Sample a ``span_length`` from the interval ``[1, max_span_length]`` (length of span of tokens to be
               masked)
            2. Reserve a context of length ``context_length = span_length / plm_probability`` to surround span to be
               masked
            3. Sample a starting point ``start_index`` from the interval ``[cur_len, cur_len + context_length -
               span_length]`` and mask tokens ``start_index:start_index + span_length``
            4. Set ``cur_len = cur_len + context_length``. If ``cur_len < max_len`` (i.e. there are tokens remaining in
               the sequence to be processed), repeat from Step 1.
        """
    if self.tokenizer.mask_token is None:
        raise ValueError(
            'This tokenizer does not have a mask token which is necessary for permutation language modeling. Please add a mask token if you want to use this tokenizer.'
            )
    if inputs.size(1) % 2 != 0:
        raise ValueError(
            'This collator requires that sequence lengths be even to create a leakage-free perm_mask. Please see relevant comments in source code for details.'
            )
    labels = inputs.clone()
    masked_indices = torch.full(labels.shape, 0, dtype=torch.bool)
    target_mapping = torch.zeros((labels.size(0), labels.size(1), labels.
        size(1)), dtype=torch.float32)
    for i in range(labels.size(0)):
        cur_len = 0
        max_len = labels.size(1)
        while cur_len < max_len:
            span_length = torch.randint(1, self.max_span_length + 1, (1,)
                ).item()
            context_length = int(span_length / self.plm_probability)
            start_index = cur_len + torch.randint(context_length -
                span_length + 1, (1,)).item()
            masked_indices[i, start_index:start_index + span_length] = 1
            cur_len += context_length
        target_mapping[i] = torch.eye(labels.size(1))
    special_tokens_mask = torch.tensor([self.tokenizer.
        get_special_tokens_mask(val, already_has_special_tokens=True) for
        val in labels.tolist()], dtype=torch.bool)
    masked_indices.masked_fill_(special_tokens_mask, value=0.0)
    if self.tokenizer._pad_token is not None:
        padding_mask = labels.eq(self.tokenizer.pad_token_id)
        masked_indices.masked_fill_(padding_mask, value=0.0)
    non_func_mask = ~(padding_mask | special_tokens_mask)
    inputs[masked_indices] = self.tokenizer.mask_token_id
    labels[~masked_indices] = -100
    perm_mask = torch.zeros((labels.size(0), labels.size(1), labels.size(1)
        ), dtype=torch.float32)
    for i in range(labels.size(0)):
        perm_index = torch.arange(labels.size(1))
        perm_index = perm_index.reshape((-1, labels.size(1) // 2)).transpose(
            0, 1)
        perm_index = perm_index[torch.randperm(labels.size(1) // 2)]
        perm_index = torch.flatten(perm_index.transpose(0, 1))
        perm_index.masked_fill_(~masked_indices[i] & non_func_mask[i], -1)
        perm_mask[i] = (perm_index.reshape((labels.size(1), 1)) <=
            perm_index.reshape((1, labels.size(1)))) & masked_indices[i]
    return inputs.long(), perm_mask, target_mapping, labels.long()

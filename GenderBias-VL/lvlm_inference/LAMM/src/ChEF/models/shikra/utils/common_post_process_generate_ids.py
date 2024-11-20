def post_process_generate_ids(tokenizer: PreTrainedTokenizer, ids: torch.Tensor
    ):
    ids = copy.deepcopy(ids)
    ids[ids < 0] = tokenizer.pad_token_id
    return ids

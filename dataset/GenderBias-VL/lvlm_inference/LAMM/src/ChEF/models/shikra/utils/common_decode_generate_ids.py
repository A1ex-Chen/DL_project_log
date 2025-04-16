def decode_generate_ids(tokenizer: PreTrainedTokenizer, ids: torch.Tensor
    ) ->Union[List[str], str]:
    assert ids.ndim in [1, 2]
    only_one_sentence = ids.ndim == 1
    if only_one_sentence:
        ids = ids.unsqueeze(0)
    ids = post_process_generate_ids(tokenizer, ids)
    res = tokenizer.batch_decode(ids, skip_special_tokens=True,
        clean_up_tokenization_spaces=True)
    if only_one_sentence:
        return res[0]
    return res

def post_process_prediction(hypo_tokens, src_str, alignment, align_dict,
    tgt_dict, remove_bpe=None, extra_symbols_to_ignore=None):
    hypo_str = tgt_dict.string(hypo_tokens, remove_bpe,
        extra_symbols_to_ignore=extra_symbols_to_ignore)
    if align_dict is not None:
        hypo_str = utils.replace_unk(hypo_str, src_str, alignment,
            align_dict, tgt_dict.unk_string())
    if align_dict is not None or remove_bpe is not None:
        hypo_tokens = tgt_dict.encode_line(hypo_str, add_if_not_exist=False)
    return hypo_tokens, hypo_str, alignment

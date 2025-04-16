@staticmethod
def _reorder_cache(past_key_values, beam_idx):
    """Used by HuggingFace generate when using beam search with kv-caching.

        See https://github.com/huggingface/transformers/blob/3ec7a47664ebe40c40f4b722f6bb1cd30c3821ec/src/transformers/models/gpt2/modeling_gpt2.py#L1122-L1133
        for an example in transformers.
        """
    reordered_past = []
    for layer_past in past_key_values:
        reordered_past += [tuple(past_state.index_select(0, beam_idx) for
            past_state in layer_past)]
    return reordered_past

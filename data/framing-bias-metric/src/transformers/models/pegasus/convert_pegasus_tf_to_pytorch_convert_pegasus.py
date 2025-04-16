def convert_pegasus(tf_weights: dict, cfg_updates: dict
    ) ->PegasusForConditionalGeneration:
    cfg_kwargs = DEFAULTS.copy()
    cfg_kwargs.update(cfg_updates)
    cfg = PegasusConfig(**cfg_kwargs)
    torch_model = PegasusForConditionalGeneration(cfg)
    sd = torch_model.model.state_dict()
    mapping = {}
    for k, v in tf_weights.items():
        new_k = rename_state_dict_key(k)
        if new_k not in sd:
            raise ValueError(
                f'could not find new key {new_k} in state dict. (converted from {k})'
                )
        if 'dense' in k or 'proj' in new_k:
            v = v.T
        mapping[new_k] = torch.tensor(v, dtype=sd[new_k].dtype)
        assert v.shape == sd[new_k
            ].shape, f'{new_k}, {k}, {v.shape}, {sd[new_k].shape}'
    mapping['shared.weight'][cfg.pad_token_id] = torch.zeros_like(mapping[
        'shared.weight'][cfg.pad_token_id + 1])
    mapping['encoder.embed_tokens.weight'] = mapping['shared.weight']
    mapping['decoder.embed_tokens.weight'] = mapping['shared.weight']
    empty_biases = {k: torch.zeros_like(v) for k, v in sd.items() if k.
        endswith('bias') and k not in mapping}
    mapping.update(**empty_biases)
    missing, extra = torch_model.model.load_state_dict(mapping, strict=False)
    unexpected_missing = [k for k in missing if k not in [
        'encoder.embed_positions.weight', 'decoder.embed_positions.weight']]
    assert unexpected_missing == [
        ], f'no matches found for the following torch keys {unexpected_missing}'
    assert extra == [], f'no matches found for the following tf keys {extra}'
    return torch_model

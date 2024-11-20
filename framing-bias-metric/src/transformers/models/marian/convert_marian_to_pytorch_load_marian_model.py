def load_marian_model(self) ->MarianMTModel:
    state_dict, cfg = self.state_dict, self.hf_config
    assert cfg.static_position_embeddings, 'config.static_position_embeddings should be True'
    model = MarianMTModel(cfg)
    assert 'hidden_size' not in cfg.to_dict()
    load_layers_(model.model.encoder.layers, state_dict, BART_CONVERTER)
    load_layers_(model.model.decoder.layers, state_dict, BART_CONVERTER,
        is_decoder=True)
    wemb_tensor = torch.nn.Parameter(torch.FloatTensor(self.wemb))
    bias_tensor = torch.nn.Parameter(torch.FloatTensor(self.final_bias))
    model.model.shared.weight = wemb_tensor
    model.model.encoder.embed_tokens = model.model.decoder.embed_tokens = (
        model.model.shared)
    model.final_logits_bias = bias_tensor
    if 'Wpos' in state_dict:
        print('Unexpected: got Wpos')
        wpos_tensor = torch.tensor(state_dict['Wpos'])
        model.model.encoder.embed_positions.weight = wpos_tensor
        model.model.decoder.embed_positions.weight = wpos_tensor
    if cfg.normalize_embedding:
        assert 'encoder_emb_ln_scale_pre' in state_dict
        raise NotImplementedError('Need to convert layernorm_embedding')
    assert not self.extra_keys, f'Failed to convert {self.extra_keys}'
    assert model.model.shared.padding_idx == self.pad_token_id, f'Padding tokens {model.model.shared.padding_idx} and {self.pad_token_id} mismatched'
    return model

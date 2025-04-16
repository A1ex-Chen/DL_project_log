def _validate_config(self):
    self.attn_config = self._set_config_defaults(self.attn_config,
        attn_config_defaults)
    self.init_config = self._set_config_defaults(self.init_config,
        init_config_defaults)
    if self.d_model % self.n_heads != 0:
        raise ValueError('d_model must be divisible by n_heads')
    if any(prob < 0 or prob > 1 for prob in [self.attn_config['attn_pdrop'],
        self.resid_pdrop, self.emb_pdrop]):
        raise ValueError(
            "self.attn_config['attn_pdrop'], resid_pdrop, emb_pdrop are probabilities and must be between 0 and 1"
            )
    if self.attn_config['attn_impl'] not in ['torch', 'flash', 'triton']:
        raise ValueError(f"Unknown attn_impl={self.attn_config['attn_impl']}")
    if self.attn_config['prefix_lm'] and self.attn_config['attn_impl'] not in [
        'torch', 'triton']:
        raise NotImplementedError(
            'prefix_lm only implemented with torch and triton attention.')
    if self.attn_config['alibi'] and self.attn_config['attn_impl'] not in [
        'torch', 'triton']:
        raise NotImplementedError(
            'alibi only implemented with torch and triton attention.')
    if self.attn_config['attn_uses_sequence_id'] and self.attn_config[
        'attn_impl'] not in ['torch', 'triton']:
        raise NotImplementedError(
            'attn_uses_sequence_id only implemented with torch and triton attention.'
            )
    if self.embedding_fraction > 1 or self.embedding_fraction <= 0:
        raise ValueError(
            'model.embedding_fraction must be between 0 (exclusive) and 1 (inclusive)!'
            )
    if isinstance(self.logit_scale, str
        ) and self.logit_scale != 'inv_sqrt_d_model':
        raise ValueError(
            f"self.logit_scale={self.logit_scale!r} is not recognized as an option; use numeric value or 'inv_sqrt_d_model'."
            )
    if self.init_config.get('name', None) is None:
        raise ValueError(
            f"self.init_config={self.init_config!r} 'name' needs to be set.")
    if not self.learned_pos_emb and not self.attn_config['alibi']:
        raise ValueError(
            f'Positional information must be provided to the model using either learned_pos_emb or alibi.'
            )

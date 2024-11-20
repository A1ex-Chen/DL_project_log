def _set_pos_net_if_use_gligen(self, attention_type: str,
    cross_attention_dim: int):
    if attention_type in ['gated', 'gated-text-image']:
        positive_len = 768
        if isinstance(cross_attention_dim, int):
            positive_len = cross_attention_dim
        elif isinstance(cross_attention_dim, tuple) or isinstance(
            cross_attention_dim, list):
            positive_len = cross_attention_dim[0]
        feature_type = ('text-only' if attention_type == 'gated' else
            'text-image')
        self.position_net = GLIGENTextBoundingboxProjection(positive_len=
            positive_len, out_dim=cross_attention_dim, feature_type=
            feature_type)

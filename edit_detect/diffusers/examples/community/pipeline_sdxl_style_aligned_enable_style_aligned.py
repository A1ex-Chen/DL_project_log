def enable_style_aligned(self, share_group_norm: bool=True,
    share_layer_norm: bool=True, share_attention: bool=True, adain_queries:
    bool=True, adain_keys: bool=True, adain_values: bool=False,
    full_attention_share: bool=False, shared_score_scale: float=1.0,
    shared_score_shift: float=0.0, only_self_level: float=0.0):
    """
        Enables the StyleAligned mechanism as in https://arxiv.org/abs/2312.02133.

        Args:
            share_group_norm (`bool`, defaults to `True`):
                Whether or not to use shared group normalization layers.
            share_layer_norm (`bool`, defaults to `True`):
                Whether or not to use shared layer normalization layers.
            share_attention (`bool`, defaults to `True`):
                Whether or not to use attention sharing between batch images.
            adain_queries (`bool`, defaults to `True`):
                Whether or not to apply the AdaIn operation on attention queries.
            adain_keys (`bool`, defaults to `True`):
                Whether or not to apply the AdaIn operation on attention keys.
            adain_values (`bool`, defaults to `False`):
                Whether or not to apply the AdaIn operation on attention values.
            full_attention_share (`bool`, defaults to `False`):
                Whether or not to use full attention sharing between all images in a batch. Can
                lead to content leakage within each batch and some loss in diversity.
            shared_score_scale (`float`, defaults to `1.0`):
                Scale for shared attention.
        """
    self._style_aligned_norm_layers = self._register_shared_norm(
        share_group_norm, share_layer_norm)
    self._enable_shared_attention_processors(share_attention=
        share_attention, adain_queries=adain_queries, adain_keys=adain_keys,
        adain_values=adain_values, full_attention_share=
        full_attention_share, shared_score_scale=shared_score_scale,
        shared_score_shift=shared_score_shift, only_self_level=only_self_level)

def init_otter(self, media_token_id: int, vis_hidden_size: int,
    cross_attn_every_n_layers: int, use_media_placement_augmentation: bool,
    only_attend_previous: bool):
    """
        Initialize Otter by adding a new gated cross attn to the decoder. Store the media token id for computing the media locations.
        """
    gated_cross_attn_layers = nn.ModuleList([(OtterGatedCrossAttentionBlock
        (dim=self.config.hidden_size, dim_visual=vis_hidden_size,
        only_attend_previous=only_attend_previous) if (layer_idx + 1) %
        cross_attn_every_n_layers == 0 else None) for layer_idx, _ in
        enumerate(self._get_decoder_layers())])
    self._set_decoder_layers(nn.ModuleList([OtterLayer(
        gated_cross_attn_layer, decoder_layer) for gated_cross_attn_layer,
        decoder_layer in zip(gated_cross_attn_layers, self.
        _get_decoder_layers())]))
    self.media_token_id = media_token_id
    self.use_media_placement_augmentation = use_media_placement_augmentation
    self.only_attend_previous = only_attend_previous
    self.initialized_otter = True

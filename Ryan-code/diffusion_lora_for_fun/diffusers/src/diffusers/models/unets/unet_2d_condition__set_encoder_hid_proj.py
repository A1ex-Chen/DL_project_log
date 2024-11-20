def _set_encoder_hid_proj(self, encoder_hid_dim_type: Optional[str],
    cross_attention_dim: Union[int, Tuple[int]], encoder_hid_dim: Optional[int]
    ):
    if encoder_hid_dim_type is None and encoder_hid_dim is not None:
        encoder_hid_dim_type = 'text_proj'
        self.register_to_config(encoder_hid_dim_type=encoder_hid_dim_type)
        logger.info(
            "encoder_hid_dim_type defaults to 'text_proj' as `encoder_hid_dim` is defined."
            )
    if encoder_hid_dim is None and encoder_hid_dim_type is not None:
        raise ValueError(
            f'`encoder_hid_dim` has to be defined when `encoder_hid_dim_type` is set to {encoder_hid_dim_type}.'
            )
    if encoder_hid_dim_type == 'text_proj':
        self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attention_dim)
    elif encoder_hid_dim_type == 'text_image_proj':
        self.encoder_hid_proj = TextImageProjection(text_embed_dim=
            encoder_hid_dim, image_embed_dim=cross_attention_dim,
            cross_attention_dim=cross_attention_dim)
    elif encoder_hid_dim_type == 'image_proj':
        self.encoder_hid_proj = ImageProjection(image_embed_dim=
            encoder_hid_dim, cross_attention_dim=cross_attention_dim)
    elif encoder_hid_dim_type is not None:
        raise ValueError(
            f"encoder_hid_dim_type: {encoder_hid_dim_type} must be None, 'text_proj' or 'text_image_proj'."
            )
    else:
        self.encoder_hid_proj = None

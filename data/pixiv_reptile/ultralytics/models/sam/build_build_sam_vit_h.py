def build_sam_vit_h(checkpoint=None):
    """Build and return a Segment Anything Model (SAM) h-size model."""
    return _build_sam(encoder_embed_dim=1280, encoder_depth=32,
        encoder_num_heads=16, encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint)

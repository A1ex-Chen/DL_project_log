def build_mobile_sam(checkpoint=None):
    """Build and return Mobile Segment Anything Model (Mobile-SAM)."""
    return _build_sam(encoder_embed_dim=[64, 128, 160, 320], encoder_depth=
        [2, 2, 6, 2], encoder_num_heads=[2, 4, 5, 10],
        encoder_global_attn_indexes=None, mobile_sam=True, checkpoint=
        checkpoint)

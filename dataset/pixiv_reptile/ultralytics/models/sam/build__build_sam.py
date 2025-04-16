def _build_sam(encoder_embed_dim, encoder_depth, encoder_num_heads,
    encoder_global_attn_indexes, checkpoint=None, mobile_sam=False):
    """Builds the selected SAM model architecture."""
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    image_encoder = TinyViT(img_size=1024, in_chans=3, num_classes=1000,
        embed_dims=encoder_embed_dim, depths=encoder_depth, num_heads=
        encoder_num_heads, window_sizes=[7, 7, 14, 7], mlp_ratio=4.0,
        drop_rate=0.0, drop_path_rate=0.0, use_checkpoint=False,
        mbconv_expand_ratio=4.0, local_conv_size=3, layer_lr_decay=0.8
        ) if mobile_sam else ImageEncoderViT(depth=encoder_depth, embed_dim
        =encoder_embed_dim, img_size=image_size, mlp_ratio=4, norm_layer=
        partial(torch.nn.LayerNorm, eps=1e-06), num_heads=encoder_num_heads,
        patch_size=vit_patch_size, qkv_bias=True, use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes, window_size=14,
        out_chans=prompt_embed_dim)
    sam = Sam(image_encoder=image_encoder, prompt_encoder=PromptEncoder(
        embed_dim=prompt_embed_dim, image_embedding_size=(
        image_embedding_size, image_embedding_size), input_image_size=(
        image_size, image_size), mask_in_chans=16), mask_decoder=
        MaskDecoder(num_multimask_outputs=3, transformer=TwoWayTransformer(
        depth=2, embedding_dim=prompt_embed_dim, mlp_dim=2048, num_heads=8),
        transformer_dim=prompt_embed_dim, iou_head_depth=3,
        iou_head_hidden_dim=256), pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375])
    if checkpoint is not None:
        checkpoint = attempt_download_asset(checkpoint)
        with open(checkpoint, 'rb') as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    sam.eval()
    return sam

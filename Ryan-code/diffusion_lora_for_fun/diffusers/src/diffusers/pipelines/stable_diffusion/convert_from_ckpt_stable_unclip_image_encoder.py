def stable_unclip_image_encoder(original_config, local_files_only=False):
    """
    Returns the image processor and clip image encoder for the img2img unclip pipeline.

    We currently know of two types of stable unclip models which separately use the clip and the openclip image
    encoders.
    """
    image_embedder_config = original_config['model']['params'][
        'embedder_config']
    sd_clip_image_embedder_class = image_embedder_config['target']
    sd_clip_image_embedder_class = sd_clip_image_embedder_class.split('.')[-1]
    if sd_clip_image_embedder_class == 'ClipImageEmbedder':
        clip_model_name = image_embedder_config.params.model
        if clip_model_name == 'ViT-L/14':
            feature_extractor = CLIPImageProcessor()
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                'openai/clip-vit-large-patch14', local_files_only=
                local_files_only)
        else:
            raise NotImplementedError(
                f'Unknown CLIP checkpoint name in stable diffusion checkpoint {clip_model_name}'
                )
    elif sd_clip_image_embedder_class == 'FrozenOpenCLIPImageEmbedder':
        feature_extractor = CLIPImageProcessor()
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            'laion/CLIP-ViT-H-14-laion2B-s32B-b79K', local_files_only=
            local_files_only)
    else:
        raise NotImplementedError(
            f'Unknown CLIP image embedder class in stable diffusion checkpoint {sd_clip_image_embedder_class}'
            )
    return feature_extractor, image_encoder

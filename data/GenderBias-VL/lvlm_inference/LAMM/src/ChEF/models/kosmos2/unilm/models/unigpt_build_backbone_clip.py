def build_backbone_clip(args, visual_model_name, visual_pretrained):
    from .vl.clip import create_model
    force_quick_gelu = False
    if 'ViT-L' in visual_model_name:
        force_quick_gelu = True
    model = create_model(visual_model_name, pretrained=visual_pretrained,
        force_quick_gelu=force_quick_gelu)
    return model

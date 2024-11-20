@classmethod
def load_image_model(cls, args, task):

    def build_backbone_clip(args, visual_model_name, visual_pretrained):
        from .vl.clip import create_model
        force_quick_gelu = False
        if 'ViT-L' in visual_model_name:
            force_quick_gelu = True
        model = create_model(visual_model_name, pretrained=
            visual_pretrained, force_quick_gelu=force_quick_gelu)
        return model
    if args.image_encoder == 'none':
        return None, None
    if args.image_encoder == 'clip':
        model = build_backbone_clip(args, args.visual_model_name, args.
            visual_pretrained)
        connector = build_connector(args, args.visual_output_dim, args.
            decoder_embed_dim)
        return model, connector
    raise NotImplementedError('Unknown model name {}'.format(args.
        image_encoder))

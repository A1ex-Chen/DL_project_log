def initialize_model(args):
    """ return model, ready to trace """
    from image_classification.resnet import build_resnet
    model = build_resnet(args.config, 'fanin', 1000, fused_se=False)
    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict({k.replace('module.', ''): v for k, v in
            state_dict.items()})
        model.load_state_dict(state_dict)
    return model.half() if args.fp16 else model

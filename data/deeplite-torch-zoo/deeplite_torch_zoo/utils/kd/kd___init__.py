def __init__(self, args=None, handle_inplace_abn=True):
    super().__init__()
    pretrained = True
    if args.kd_model_checkpoint is not None:
        pretrained = False
    model_kd = get_model(model_name=args.kd_model_name, dataset_name=
        'imagenet', pretrained=pretrained, num_classes=args.num_classes)
    if args.kd_model_checkpoint is not None:
        model_kd.load_state_dict(torch.load(args.kd_model_checkpoint))
    model_kd.cpu().eval()
    if INPLACE_ABN_INSTALLED and handle_inplace_abn:
        model_kd = inplaceABN_to_ABN(model_kd)
    model_kd = fuse_bn2d_bn1d_abn(model_kd)
    self.model = model_kd.cuda().eval()
    self.mean_model_kd = None
    self.std_model_kd = None
    if hasattr(model_kd, 'default_cfg'):
        self.mean_model_kd = model_kd.default_cfg['mean']
        self.std_model_kd = model_kd.default_cfg['std']

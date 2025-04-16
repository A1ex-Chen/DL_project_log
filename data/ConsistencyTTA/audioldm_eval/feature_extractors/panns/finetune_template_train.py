def train(args):
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    pretrained_checkpoint_path = args.pretrained_checkpoint_path
    freeze_base = args.freeze_base
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    classes_num = config.classes_num
    pretrain = True if pretrained_checkpoint_path else False
    Model = eval(model_type)
    model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
        classes_num, freeze_base)
    if pretrain:
        logging.info('Load pretrained model from {}'.format(
            pretrained_checkpoint_path))
        model.load_from_pretrain(pretrained_checkpoint_path)
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)
    if 'cuda' in device:
        model.to(device)
    print('Load pretrained model successfully!')

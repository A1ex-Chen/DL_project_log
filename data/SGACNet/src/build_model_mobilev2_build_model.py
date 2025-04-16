def build_model(args, n_classes):
    if (not args.pretrained_on_imagenet or args.last_ckpt or args.
        pretrained_scenenet != ''):
        pretrained_on_imagenet = False
    else:
        pretrained_on_imagenet = True
    if 'decreasing' in args.decoder_channels_mode:
        if args.decoder_channels_mode == 'decreasing':
            channels_decoder = [512, 256, 128]
        warnings.warn(
            'Argument --channels_decoder is ignored when --decoder_chanels_mode decreasing is set.'
            )
    else:
        channels_decoder = [args.channels_decoder] * 3
    if isinstance(args.nr_decoder_blocks, int):
        nr_decoder_blocks = [args.nr_decoder_blocks] * 3
    elif len(args.nr_decoder_blocks) == 1:
        nr_decoder_blocks = args.nr_decoder_blocks * 3
    else:
        nr_decoder_blocks = args.nr_decoder_blocks
        assert len(nr_decoder_blocks) == 3
    if args.modality == 'rgbd':
        if args.encoder_depth in [None, 'None']:
            args.encoder_depth = args.encoder
        model = SGACNet(height=args.height, width=args.width, num_classes=
            n_classes, pretrained_on_imagenet=pretrained_on_imagenet,
            pretrained_dir=args.pretrained_dir, encoder_rgb=args.encoder,
            encoder_depth=args.encoder_depth, encoder_block=args.
            encoder_block, activation=args.activation,
            encoder_decoder_fusion=args.encoder_decoder_fusion,
            context_module=args.context_module, nr_decoder_blocks=
            nr_decoder_blocks, channels_decoder=channels_decoder,
            fuse_depth_in_rgb_encoder=args.fuse_depth_in_rgb_encoder,
            upsampling=args.upsampling)
    else:
        if args.modality == 'rgb':
            input_channels = 3
        else:
            input_channels = 1
        model = SGACNetOneModality(height=args.height, width=args.width,
            pretrained_on_imagenet=pretrained_on_imagenet, encoder=args.
            encoder, encoder_block=args.encoder_block, activation=args.
            activation, input_channels=input_channels,
            encoder_decoder_fusion=args.encoder_decoder_fusion,
            context_module=args.context_module, num_classes=n_classes,
            pretrained_dir=args.pretrained_dir, nr_decoder_blocks=
            nr_decoder_blocks, channels_decoder=channels_decoder,
            weighting_in_encoder=args.fuse_depth_in_rgb_encoder, upsampling
            =args.upsampling)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print('Device:', device)
    model.to(device)
    print(model)
    if args.he_init:
        module_list = []
        for c in model.children():
            if pretrained_on_imagenet and isinstance(c, MobileNetV2):
                continue
            for m in c.modules():
                module_list.append(m)
        for i, m in enumerate(module_list):
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                if m.out_channels == n_classes or isinstance(module_list[i +
                    1], nn.Sigmoid) or m.groups == m.in_channels:
                    continue
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        print('Applied He init.')
    if args.pretrained_scenenet != '':
        checkpoint = torch.load(args.pretrained_scenenet)
        weights_scenenet = checkpoint['state_dict']
        keys_to_ignore = [k for k in weights_scenenet if 'out' in k or 
            'decoder.upsample1' in k or 'decoder.upsample2' in k]
        if args.context_module not in ['ppm', 'appm']:
            keys_to_ignore.extend([k for k in weights_scenenet if 
                'context_module.features' in k])
        for key in keys_to_ignore:
            weights_scenenet.pop(key)
        weights_model = model.state_dict()
        weights_model.update(weights_scenenet)
        model.load_state_dict(weights_model)
        print(f'Loaded pretrained SceneNet weights: {args.pretrained_scenenet}'
            )
    if args.finetune is not None:
        checkpoint = torch.load(args.finetune)
        model.load_state_dict(checkpoint['state_dict'])
        print(f'Loaded weights for finetuning: {args.finetune}')
        print('Freeze the encoder(s).')
        for name, param in model.named_parameters():
            if ('encoder_rgb' in name or 'encoder_depth' in name or 
                'se_layer' in name):
                param.requires_grad = False
    return model, device

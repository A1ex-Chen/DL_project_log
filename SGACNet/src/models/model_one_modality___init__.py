def __init__(self, height=480, width=640, num_classes=37, encoder=
    'resnet18', encoder_block='BasicBlock', channels_decoder=None,
    pretrained_on_imagenet=True, pretrained_dir=
    '/results_nas/moko3016/moko3016-efficient-rgbd-segmentation/imagenet_pretraining'
    , activation='relu', input_channels=3, encoder_decoder_fusion='add',
    context_module='ppm', nr_decoder_blocks=None, weighting_in_encoder=
    'None', upsampling='bilinear'):
    super(SGACNetOneModality, self).__init__()
    if channels_decoder is None:
        channels_decoder = [128, 128, 128]
    if nr_decoder_blocks is None:
        nr_decoder_blocks = [1, 1, 1]
    self.weighting_in_encoder = weighting_in_encoder
    if activation.lower() == 'relu':
        self.activation = nn.ReLU(inplace=True)
    elif activation.lower() in ['swish', 'silu']:
        self.activation = Swish()
    elif activation.lower() == 'hswish':
        self.activation = Hswish()
    else:
        raise NotImplementedError(
            'Only relu, swish and hswish as activation function are supported so far. Got {}'
            .format(activation))
    if encoder == 'resnet18':
        self.encoder = ResNet18(block=encoder_block, pretrained_on_imagenet
            =pretrained_on_imagenet, pretrained_dir=pretrained_dir,
            activation=self.activation, input_channels=input_channels)
    elif encoder == 'resnet34':
        self.encoder = ResNet34(block=encoder_block, pretrained_on_imagenet
            =pretrained_on_imagenet, pretrained_dir=pretrained_dir,
            activation=self.activation, input_channels=input_channels)
    elif encoder == 'resnet50':
        self.encoder = ResNet50(pretrained_on_imagenet=
            pretrained_on_imagenet, activation=self.activation,
            input_channels=input_channels)
    else:
        raise NotImplementedError(
            'Only ResNets as encoder are supported so far. Got {}'.format(
            activation))
    self.channels_decoder_in = self.encoder.down_32_channels_out
    if weighting_in_encoder == 'SE-add':
        self.se_layer0 = SqueezeAndExcitation(64, activation=self.activation)
        self.se_layer1 = SqueezeAndExcitation(self.encoder.
            down_4_channels_out, activation=self.activation)
        self.se_layer2 = SqueezeAndExcitation(self.encoder.
            down_8_channels_out, activation=self.activation)
        self.se_layer3 = SqueezeAndExcitation(self.encoder.
            down_16_channels_out, activation=self.activation)
        self.se_layer4 = SqueezeAndExcitation(self.encoder.
            down_32_channels_out, activation=self.activation)
    else:
        self.se_layer0 = nn.Identity()
        self.se_layer1 = nn.Identity()
        self.se_layer2 = nn.Identity()
        self.se_layer3 = nn.Identity()
        self.se_layer4 = nn.Identity()
    if encoder_decoder_fusion == 'add':
        layers_skip1 = list()
        if self.encoder.down_4_channels_out != channels_decoder[2]:
            layers_skip1.append(ConvBNAct(self.encoder.down_4_channels_out,
                channels_decoder[2], kernel_size=1, activation=self.activation)
                )
        self.skip_layer1 = nn.Sequential(*layers_skip1)
        layers_skip2 = list()
        if self.encoder.down_8_channels_out != channels_decoder[1]:
            layers_skip2.append(ConvBNAct(self.encoder.down_8_channels_out,
                channels_decoder[1], kernel_size=1, activation=self.activation)
                )
        self.skip_layer2 = nn.Sequential(*layers_skip2)
        layers_skip3 = list()
        if self.encoder.down_16_channels_out != channels_decoder[0]:
            layers_skip3.append(ConvBNAct(self.encoder.down_16_channels_out,
                channels_decoder[0], kernel_size=1, activation=self.activation)
                )
        self.skip_layer3 = nn.Sequential(*layers_skip3)
    if 'learned-3x3' in upsampling:
        warnings.warn(
            'for the context module the learned upsampling is not possible as the feature maps are not upscaled by the factor 2. We will use nearest neighbor instead.'
            )
        upsampling_context_module = 'nearest'
    else:
        upsampling_context_module = upsampling
    self.context_module, channels_after_context_module = get_context_module(
        context_module, self.channels_decoder_in, channels_decoder[0],
        input_size=(height // 32, width // 32), activation=self.activation,
        upsampling_mode=upsampling_context_module)
    self.decoder = Decoder(channels_in=channels_after_context_module,
        channels_decoder=channels_decoder, activation=self.activation,
        nr_decoder_blocks=nr_decoder_blocks, encoder_decoder_fusion=
        encoder_decoder_fusion, upsampling_mode=upsampling, num_classes=
        num_classes)

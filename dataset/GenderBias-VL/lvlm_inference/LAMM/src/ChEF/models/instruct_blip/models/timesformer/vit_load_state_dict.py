def load_state_dict(self, pretrained_ckpt_path):
    logging.info('Loading TimeSformer checkpoints from {}'.format(
        pretrained_ckpt_path))
    if pretrained_ckpt_path == 'vit_base_patch16_224':
        load_ckpt_func = load_pretrained_imagenet
    else:
        load_ckpt_func = load_pretrained_kinetics
    load_ckpt_func(self.model, num_classes=self.model.num_classes, in_chans
        =3, filter_fn=_conv_filter, img_size=self.img_size, num_frames=self
        .num_frames, num_patches=self.num_patches, attention_type=self.
        attention_type, pretrained_model=pretrained_ckpt_path)

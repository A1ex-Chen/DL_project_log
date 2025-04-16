def load_pretrained(model, cfg=None, num_classes=1000, in_chans=3,
    filter_fn=None, img_size=224, num_frames=8, num_patches=196,
    attention_type='divided_space_time', pretrained_model='', strict=True):
    if cfg is None:
        cfg = getattr(model, 'default_cfg')
    if cfg is None or 'url' not in cfg or not cfg['url']:
        logging.warning(
            'Pretrained model URL is invalid, using random initialization.')
        return
    if len(pretrained_model) == 0:
        if cfg is None:
            logging.info(f'loading from default config {model.default_cfg}.')
        state_dict = model_zoo.load_url(cfg['url'], progress=False,
            map_location='cpu')
    else:
        try:
            state_dict = load_state_dict(pretrained_model)['model']
        except:
            state_dict = load_state_dict(pretrained_model)
    if filter_fn is not None:
        state_dict = filter_fn(state_dict)
    if in_chans == 1:
        conv1_name = cfg['first_conv']
        logging.info(
            'Converting first conv (%s) pretrained weights from 3 to 1 channel'
             % conv1_name)
        conv1_weight = state_dict[conv1_name + '.weight']
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I > 3:
            assert conv1_weight.shape[1] % 3 == 0
            conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
        else:
            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
        conv1_weight = conv1_weight.to(conv1_type)
        state_dict[conv1_name + '.weight'] = conv1_weight
    elif in_chans != 3:
        conv1_name = cfg['first_conv']
        conv1_weight = state_dict[conv1_name + '.weight']
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I != 3:
            logging.warning(
                'Deleting first conv (%s) from pretrained weights.' %
                conv1_name)
            del state_dict[conv1_name + '.weight']
            strict = False
        else:
            logging.info(
                'Repeating first conv (%s) weights in channel dim.' %
                conv1_name)
            repeat = int(math.ceil(in_chans / 3))
            conv1_weight = conv1_weight.repeat(1, repeat, 1, 1)[:, :
                in_chans, :, :]
            conv1_weight *= 3 / float(in_chans)
            conv1_weight = conv1_weight.to(conv1_type)
            state_dict[conv1_name + '.weight'] = conv1_weight
    classifier_name = cfg['classifier']
    if num_classes == 1000 and cfg['num_classes'] == 1001:
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:]
    elif num_classes != state_dict[classifier_name + '.weight'].size(0):
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False
    logging.info(
        f"Resizing spatial position embedding from {state_dict['pos_embed'].size(1)} to {num_patches + 1}"
        )
    if num_patches + 1 != state_dict['pos_embed'].size(1):
        pos_embed = state_dict['pos_embed']
        cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
        other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
        new_pos_embed = F.interpolate(other_pos_embed, size=num_patches,
            mode='nearest')
        new_pos_embed = new_pos_embed.transpose(1, 2)
        new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
        state_dict['pos_embed'] = new_pos_embed
    if 'time_embed' in state_dict and num_frames != state_dict['time_embed'
        ].size(1):
        logging.info(
            f"Resizing temporal position embedding from {state_dict['time_embed'].size(1)} to {num_frames}"
            )
        time_embed = state_dict['time_embed'].transpose(1, 2)
        new_time_embed = F.interpolate(time_embed, size=num_frames, mode=
            'nearest')
        state_dict['time_embed'] = new_time_embed.transpose(1, 2)
    if attention_type == 'divided_space_time':
        new_state_dict = state_dict.copy()
        for key in state_dict:
            if 'blocks' in key and 'attn' in key:
                new_key = key.replace('attn', 'temporal_attn')
                if not new_key in state_dict:
                    new_state_dict[new_key] = state_dict[key]
                else:
                    new_state_dict[new_key] = state_dict[new_key]
            if 'blocks' in key and 'norm1' in key:
                new_key = key.replace('norm1', 'temporal_norm1')
                if not new_key in state_dict:
                    new_state_dict[new_key] = state_dict[key]
                else:
                    new_state_dict[new_key] = state_dict[new_key]
        state_dict = new_state_dict
    model.load_state_dict(state_dict, strict=False)

def load_pretrained_with_different_encoder_block(model, encoder_block,
    input_channels, resnet_name, pretrained_dir='./trained_models/imagenet'):
    ckpt_path = os.path.join(pretrained_dir, f'{resnet_name}_NBt1D.pth')
    if not os.path.exists(ckpt_path):
        logs = pd.read_csv(os.path.join(pretrained_dir, 'logs.csv'))
        idx_top1 = logs['acc_val_top-1'].idxmax()
        acc_top1 = logs['acc_val_top-1'][idx_top1]
        epoch = logs.epoch[idx_top1]
        ckpt_path = os.path.join(pretrained_dir, 'ckpt_epoch_{}.pth'.format
            (epoch))
        print(f'Choosing checkpoint {ckpt_path} with top1 acc {acc_top1}')
    if torch.cuda.is_available():
        checkpoint = torch.load(ckpt_path)
    else:
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    checkpoint['state_dict2'] = OrderedDict()
    for key in checkpoint['state_dict']:
        if 'encoder' in key:
            checkpoint['state_dict2'][key.split('encoder.')[-1]] = checkpoint[
                'state_dict'][key]
    weights = checkpoint['state_dict2']
    if input_channels == 1:
        weights['conv1.weight'] = torch.sum(weights['conv1.weight'], axis=1,
            keepdim=True)
    model.load_state_dict(weights, strict=False)
    print(
        f'Loaded {resnet_name} with encoder block {encoder_block} pretrained on ImageNet'
        )
    print(ckpt_path)
    return model

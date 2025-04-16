def fuse_bn2d_bn1d_abn(model):
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            next_bn = compute_next_bn(n, model)
            if next_bn is not None:
                next_bn_ = extract_layer(model, next_bn)
                fuse_bn_to_conv(next_bn_, m)
                set_layer(model, next_bn, nn.Identity())
            next_abn = compute_next_abn(n, model)
            if next_abn is not None:
                next_bn_ = extract_layer(model, next_abn)
                activation = calc_abn_activation(next_bn_)
                fuse_bn_to_conv(next_bn_, m)
                set_layer(model, next_abn, activation)
        if isinstance(m, torch.nn.Linear):
            next_bn1d = compute_next_bn_1d(n, model)
            if next_bn1d is not None:
                next_bn1d_ = extract_layer(model, next_bn1d)
                fuse_bn_to_linear(next_bn1d_, m)
                set_layer(model, next_bn1d, nn.Identity())
    return model

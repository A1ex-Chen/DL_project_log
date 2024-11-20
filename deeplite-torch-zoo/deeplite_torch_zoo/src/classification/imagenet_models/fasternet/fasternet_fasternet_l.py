def fasternet_l(num_classes=1000):
    model = FasterNet(num_classes=num_classes, mlp_ratio=2, embed_dim=192,
        depths=(3, 4, 18, 3), feature_dim=1280, patch_size=4, patch_stride=
        4, patch_size2=2, patch_stride2=2, layer_scale_init_value=0.0,
        drop_path_rate=0.3, act='relu', n_div=4)
    return model

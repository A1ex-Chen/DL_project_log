def available_models():
    models = {m.name: m for m in [resnet50, resnext101_32x4d,
        se_resnext101_32x4d, efficientnet_b0, efficientnet_b4,
        efficientnet_widese_b0, efficientnet_widese_b4]}
    return models

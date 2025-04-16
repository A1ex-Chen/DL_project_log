def build_model(cfg, num_classes, device):
    model = Model(cfg, channels=3, num_classes=num_classes).to(device)
    return model

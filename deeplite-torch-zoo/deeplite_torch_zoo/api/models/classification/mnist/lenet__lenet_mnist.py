def _lenet_mnist(arch, pretrained=False, num_classes=10):
    model = LeNet5(output=num_classes)
    if pretrained:
        checkpoint_url = model_urls[arch]
        model = load_pretrained_weights(model, checkpoint_url)
    return model

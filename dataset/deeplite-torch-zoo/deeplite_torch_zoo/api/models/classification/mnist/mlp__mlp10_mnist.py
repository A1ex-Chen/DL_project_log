def _mlp10_mnist(arch, n_hiddens, pretrained=False, num_classes=10):
    model = MLP(input_dims=784, n_hiddens=n_hiddens, n_class=num_classes)
    if pretrained:
        checkpoint_url = model_urls[arch]
        model = load_pretrained_weights(model, checkpoint_url)
    return model

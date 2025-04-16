def extract_layers(model):
    list_layers = []
    for n, _ in model.named_modules():
        list_layers.append(n)
    return list_layers

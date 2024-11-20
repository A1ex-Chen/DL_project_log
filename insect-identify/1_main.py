def main():
    """
    model = getattr(models, 'vgg16')(pretrained=True)
    print(model.parameters())
    print(model.eval())
    feature = list(model.features)[:30]
    for layer in feature[:25]:
        for param in layer.parameters():
            param.requires_grad = False

    for k,v in model.named_parameters():
        print('{}: {}'.format(k, v.requires_grad))
    """
    """
    model = getattr(models, 'resnet50')(pretrained=True)
    #print(model.eval())
    for module in model.children():
        print(module)
    frozen = 0
    for child in model.children():
        frozen += 1
        if frozen < 8:
            for param in child.parameters():
                param.requires_grad = False

    for k,v in model.named_parameters():
        print('{}: {}'.format(k, v.requires_grad))
"""
    model = getattr(models, 'resnet50')(pretrained=True)
    model_dict = model.state_dict()
    print(model.conv1.weight)
    premodel_dict = torch.load(
        'D:\\Desktop\\A-classifier-with-PyTorch-master\\A-classifier-with-PyTorch-master\\checkpoint_dir\\resnet50_0.497.pkl'
        )
    premodel_dict = {k: v for k, v in premodel_dict.items() if k in
        model_dict and 'fc' not in k}
    model_dict.update(premodel_dict)
    model.load_state_dict(model_dict)
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())
    print(model.conv1.weight)
    print(model.eval())

def load_classifier(name='resnet101', n=2):
    model = torchvision.models.__dict__[name](pretrained=True)
    filters = model.fc.weight.shape[1]
    model.fc.bias = nn.Parameter(torch.zeros(n), requires_grad=True)
    model.fc.weight = nn.Parameter(torch.zeros(n, filters), requires_grad=True)
    model.fc.out_features = n
    return model

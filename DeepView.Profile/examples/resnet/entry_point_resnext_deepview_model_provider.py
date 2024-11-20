def deepview_model_provider():
    return resnet.resnext50_32x4d().cuda()

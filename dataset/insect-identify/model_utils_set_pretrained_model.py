def set_pretrained_model(model_name):
    """ Load the model to use
		Freeze the features parameters to avoid backprop through them
		returns the model
	"""
    model = getattr(models, model_name)(pretrained=True)
    model_dict = model.state_dict()
    premodel_dict = torch.load(
        'D:\\Desktop\\A-classifier-with-PyTorch-master\\A-classifier-with-PyTorch-master\\checkpoint_dir\\resnet50_0.497.pkl'
        )
    premodel_dict = {k: v for k, v in premodel_dict.items() if k in
        model_dict and 'fc' not in k}
    model_dict.update(premodel_dict)
    model.load_state_dict(model_dict)
    """
    feature = list(model.features)[:30]
	for layer in feature[:27]:
		for param in layer.parameters():
			param.requires_grad = False
    """
    return model

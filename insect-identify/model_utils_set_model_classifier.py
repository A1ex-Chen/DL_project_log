def set_model_classifier(model, hidden_layer, input_size=25088, output_size
    =15, dropout=0.5):
    """ Replace the given model classifier with the one using the specified parameters
	"""
    model.classifier = nn.Sequential(nn.Linear(input_size, hidden_layer),
        nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_layer, output_size
        ), nn.LogSoftmax(dim=1))
    return model

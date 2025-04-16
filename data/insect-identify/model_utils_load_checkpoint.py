def load_checkpoint(checkpoint_file_path):
    """ Load a checkpoint file and return the model
    """
    checkpoint = torch.load(checkpoint_file_path)
    model = getattr(models, checkpoint['model_name'])(pretrained=True)
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    return model

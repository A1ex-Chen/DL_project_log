def get_num_params(model):
    """Return the total number of parameters in a YOLO model."""
    return sum(x.numel() for x in model.parameters())

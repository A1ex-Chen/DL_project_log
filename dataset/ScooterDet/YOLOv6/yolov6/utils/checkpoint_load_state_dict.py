def load_state_dict(weights, model, map_location=None):
    """Load weights from checkpoint file, only assign weights those layers' name and shape are match."""
    ckpt = torch.load(weights, map_location=map_location)
    state_dict = ckpt['model'].float().state_dict()
    model_state_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in
        model_state_dict and v.shape == model_state_dict[k].shape}
    model.load_state_dict(state_dict, strict=False)
    del ckpt, state_dict, model_state_dict
    return model

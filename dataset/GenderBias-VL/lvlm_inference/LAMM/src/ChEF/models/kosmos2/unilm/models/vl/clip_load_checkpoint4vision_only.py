def load_checkpoint4vision_only(model, checkpoint_path, strict=True):
    state_dict = load_state_dict(checkpoint_path)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys

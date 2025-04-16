def load_pretrained_weights(model, checkpoint_url, device='cpu'):
    pretrained_dict = load_state_dict_from_url(checkpoint_url, progress=
        True, check_hash=True, map_location=device)
    load_state_dict_partial(model, pretrained_dict)
    return model

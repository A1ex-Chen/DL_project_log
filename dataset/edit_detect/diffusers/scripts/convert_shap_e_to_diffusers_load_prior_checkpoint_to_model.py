def load_prior_checkpoint_to_model(checkpoint, model):
    with tempfile.NamedTemporaryFile() as file:
        torch.save(checkpoint, file.name)
        del checkpoint
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(
            file.name), strict=False)
        missing_keys = list(set(missing_keys) - set(
            PRIOR_EXPECTED_MISSING_KEYS))
        if len(unexpected_keys) > 0:
            raise ValueError(
                f'Unexpected keys when loading prior model: {unexpected_keys}')
        if len(missing_keys) > 0:
            raise ValueError(
                f'Missing keys when loading prior model: {missing_keys}')

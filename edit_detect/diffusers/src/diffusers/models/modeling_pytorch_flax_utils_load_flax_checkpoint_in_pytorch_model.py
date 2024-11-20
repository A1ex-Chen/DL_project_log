def load_flax_checkpoint_in_pytorch_model(pt_model, model_file):
    try:
        with open(model_file, 'rb') as flax_state_f:
            flax_state = from_bytes(None, flax_state_f.read())
    except UnpicklingError as e:
        try:
            with open(model_file) as f:
                if f.read().startswith('version'):
                    raise OSError(
                        'You seem to have cloned a repository without having git-lfs installed. Please install git-lfs and run `git lfs install` followed by `git lfs pull` in the folder you cloned.'
                        )
                else:
                    raise ValueError from e
        except (UnicodeDecodeError, ValueError):
            raise EnvironmentError(
                f'Unable to convert {model_file} to Flax deserializable object. '
                )
    return load_flax_weights_in_pytorch_model(pt_model, flax_state)

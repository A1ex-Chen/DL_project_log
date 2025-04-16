def load_state_dict(checkpoint_file: Union[str, os.PathLike], variant:
    Optional[str]=None):
    """
    Reads a checkpoint file, returning properly formatted errors if they arise.
    """
    try:
        if os.path.basename(checkpoint_file) == _add_variant(WEIGHTS_NAME,
            variant):
            return torch.load(checkpoint_file, map_location='cpu')
        else:
            return safetensors.torch.load_file(checkpoint_file, device='cpu')
    except Exception as e:
        try:
            with open(checkpoint_file) as f:
                if f.read().startswith('version'):
                    raise OSError(
                        'You seem to have cloned a repository without having git-lfs installed. Please install git-lfs and run `git lfs install` followed by `git lfs pull` in the folder you cloned.'
                        )
                else:
                    raise ValueError(
                        f'Unable to locate the file {checkpoint_file} which is necessary to load this pretrained model. Make sure you have saved the model properly.'
                        ) from e
        except (UnicodeDecodeError, ValueError):
            raise OSError(
                f"Unable to load weights from checkpoint file for '{checkpoint_file}' at '{checkpoint_file}'. If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True."
                )

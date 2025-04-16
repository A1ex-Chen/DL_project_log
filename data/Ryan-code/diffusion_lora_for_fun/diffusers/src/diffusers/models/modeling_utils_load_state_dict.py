def load_state_dict(checkpoint_file: Union[str, os.PathLike], variant:
    Optional[str]=None):
    """
    Reads a checkpoint file, returning properly formatted errors if they arise.
    """
    try:
        file_extension = os.path.basename(checkpoint_file).split('.')[-1]
        if file_extension == SAFETENSORS_FILE_EXTENSION:
            return safetensors.torch.load_file(checkpoint_file, device='cpu')
        else:
            weights_only_kwarg = {'weights_only': True} if is_torch_version(
                '>=', '1.13') else {}
            return torch.load(checkpoint_file, map_location='cpu', **
                weights_only_kwarg)
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
                f"Unable to load weights from checkpoint file for '{checkpoint_file}' at '{checkpoint_file}'. "
                )

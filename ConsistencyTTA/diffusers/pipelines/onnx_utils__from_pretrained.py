@classmethod
def _from_pretrained(cls, model_id: Union[str, Path], use_auth_token:
    Optional[Union[bool, str, None]]=None, revision: Optional[Union[str,
    None]]=None, force_download: bool=False, cache_dir: Optional[str]=None,
    file_name: Optional[str]=None, provider: Optional[str]=None,
    sess_options: Optional['ort.SessionOptions']=None, **kwargs):
    """
        Load a model from a directory or the HF Hub.

        Arguments:
            model_id (`str` or `Path`):
                Directory from which to load
            use_auth_token (`str` or `bool`):
                Is needed to load models from a private or gated repository
            revision (`str`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id
            cache_dir (`Union[str, Path]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            file_name(`str`):
                Overwrites the default model file name from `"model.onnx"` to `file_name`. This allows you to load
                different model files from the same repository or directory.
            provider(`str`):
                The ONNX runtime provider, e.g. `CPUExecutionProvider` or `CUDAExecutionProvider`.
            kwargs (`Dict`, *optional*):
                kwargs will be passed to the model during initialization
        """
    model_file_name = file_name if file_name is not None else ONNX_WEIGHTS_NAME
    if os.path.isdir(model_id):
        model = OnnxRuntimeModel.load_model(os.path.join(model_id,
            model_file_name), provider=provider, sess_options=sess_options)
        kwargs['model_save_dir'] = Path(model_id)
    else:
        model_cache_path = hf_hub_download(repo_id=model_id, filename=
            model_file_name, use_auth_token=use_auth_token, revision=
            revision, cache_dir=cache_dir, force_download=force_download)
        kwargs['model_save_dir'] = Path(model_cache_path).parent
        kwargs['latest_model_name'] = Path(model_cache_path).name
        model = OnnxRuntimeModel.load_model(model_cache_path, provider=
            provider, sess_options=sess_options)
    return cls(model=model, **kwargs)

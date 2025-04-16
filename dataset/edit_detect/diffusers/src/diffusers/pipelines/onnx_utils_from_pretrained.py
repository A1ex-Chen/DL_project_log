@classmethod
@validate_hf_hub_args
def from_pretrained(cls, model_id: Union[str, Path], force_download: bool=
    True, token: Optional[str]=None, cache_dir: Optional[str]=None, **
    model_kwargs):
    revision = None
    if len(str(model_id).split('@')) == 2:
        model_id, revision = model_id.split('@')
    return cls._from_pretrained(model_id=model_id, revision=revision,
        cache_dir=cache_dir, force_download=force_download, token=token, **
        model_kwargs)

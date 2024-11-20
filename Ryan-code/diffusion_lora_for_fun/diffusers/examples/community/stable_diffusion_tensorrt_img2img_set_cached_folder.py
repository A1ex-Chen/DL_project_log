@classmethod
@validate_hf_hub_args
def set_cached_folder(cls, pretrained_model_name_or_path: Optional[Union[
    str, os.PathLike]], **kwargs):
    cache_dir = kwargs.pop('cache_dir', None)
    resume_download = kwargs.pop('resume_download', False)
    proxies = kwargs.pop('proxies', None)
    local_files_only = kwargs.pop('local_files_only', False)
    token = kwargs.pop('token', None)
    revision = kwargs.pop('revision', None)
    cls.cached_folder = pretrained_model_name_or_path if os.path.isdir(
        pretrained_model_name_or_path) else snapshot_download(
        pretrained_model_name_or_path, cache_dir=cache_dir, resume_download
        =resume_download, proxies=proxies, local_files_only=
        local_files_only, token=token, revision=revision)

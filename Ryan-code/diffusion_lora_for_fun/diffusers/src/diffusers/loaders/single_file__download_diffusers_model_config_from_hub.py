def _download_diffusers_model_config_from_hub(pretrained_model_name_or_path,
    cache_dir, revision, proxies, force_download=None, resume_download=None,
    local_files_only=None, token=None):
    allow_patterns = ['**/*.json', '*.json', '*.txt', '**/*.txt']
    cached_model_path = snapshot_download(pretrained_model_name_or_path,
        cache_dir=cache_dir, revision=revision, proxies=proxies,
        force_download=force_download, resume_download=resume_download,
        local_files_only=local_files_only, token=token, allow_patterns=
        allow_patterns)
    return cached_model_path

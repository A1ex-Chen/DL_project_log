def load_single_file_checkpoint(pretrained_model_link_or_path,
    resume_download=False, force_download=False, proxies=None, token=None,
    cache_dir=None, local_files_only=None, revision=None):
    if os.path.isfile(pretrained_model_link_or_path):
        pretrained_model_link_or_path = pretrained_model_link_or_path
    else:
        repo_id, weights_name = _extract_repo_id_and_weights_name(
            pretrained_model_link_or_path)
        pretrained_model_link_or_path = _get_model_file(repo_id,
            weights_name=weights_name, force_download=force_download,
            cache_dir=cache_dir, resume_download=resume_download, proxies=
            proxies, local_files_only=local_files_only, token=token,
            revision=revision)
    checkpoint = load_state_dict(pretrained_model_link_or_path)
    while 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    return checkpoint

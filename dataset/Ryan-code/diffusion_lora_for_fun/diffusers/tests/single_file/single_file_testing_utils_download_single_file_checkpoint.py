def download_single_file_checkpoint(repo_id, filename, tmpdir):
    path = hf_hub_download(repo_id, filename=filename, local_dir=tmpdir)
    return path

def download_diffusers_config(repo_id, tmpdir):
    path = snapshot_download(repo_id, ignore_patterns=['**/*.ckpt',
        '*.ckpt', '**/*.bin', '*.bin', '**/*.pt', '*.pt',
        '**/*.safetensors', '*.safetensors'], allow_patterns=['**/*.json',
        '*.json', '*.txt', '**/*.txt'], local_dir=tmpdir)
    return path

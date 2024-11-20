def load_from_hf(repo_id, filename, subfolder=None):
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename,
        subfolder=subfolder)
    return torch.load(cache_file, map_location='cpu')

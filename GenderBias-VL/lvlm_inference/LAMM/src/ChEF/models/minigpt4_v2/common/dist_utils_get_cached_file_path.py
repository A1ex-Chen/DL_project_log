def get_cached_file_path():
    parts = torch.hub.urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(timm_hub.get_cache_dir(), filename)
    return cached_file

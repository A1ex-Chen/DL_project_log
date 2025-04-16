def download_cached_file(url, check_hash=True, progress=False):
    """
    Download a file from a URL and cache it locally. If the file already exists, it is not downloaded again.
    If distributed, only the main process downloads the file, and the other processes wait for the file to be downloaded.
    """

    def get_cached_file_path():
        parts = torch.hub.urlparse(url)
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(timm_hub.get_cache_dir(), filename)
        return cached_file
    if is_main_process():
        timm_hub.download_cached_file(url, check_hash, progress)
    if is_dist_avail_and_initialized():
        dist.barrier()
    return get_cached_file_path()

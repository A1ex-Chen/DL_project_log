def fetch_file(link, subdir, unpack=False, md5_hash=None):
    """Convert URL to file path and download the file
    if it is not already present in spedified cache.

    Parameters
    ----------
    link : link path
        URL of the file to download
    subdir : directory path
        Local path to check for cached file.
    unpack : boolean
        Flag to specify if the file to download should
        be decompressed too.
        (default: False, no decompression)
    md5_hash : MD5 hash
        Hash used as a checksum to verify data integrity.
        Verification is carried out if a hash is provided.
        (default: None, no verification)

    Return
    ----------
    local path to the downloaded, or cached, file.
    """
    fname = os.path.basename(link)
    return get_file(fname, origin=link, unpack=unpack, md5_hash=md5_hash,
        cache_subdir=subdir)

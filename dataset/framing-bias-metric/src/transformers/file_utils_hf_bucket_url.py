def hf_bucket_url(model_id: str, filename: str, subfolder: Optional[str]=
    None, revision: Optional[str]=None, mirror=None) ->str:
    """
    Resolve a model identifier, a file name, and an optional revision id, to a huggingface.co-hosted url, redirecting
    to Cloudfront (a Content Delivery Network, or CDN) for large files.

    Cloudfront is replicated over the globe so downloads are way faster for the end user (and it also lowers our
    bandwidth costs).

    Cloudfront aggressively caches files by default (default TTL is 24 hours), however this is not an issue here
    because we migrated to a git-based versioning system on huggingface.co, so we now store the files on S3/Cloudfront
    in a content-addressable way (i.e., the file name is its hash). Using content-addressable filenames means cache
    can't ever be stale.

    In terms of client-side caching from this library, we base our caching on the objects' ETag. An object' ETag is:
    its sha1 if stored in git, or its sha256 if stored in git-lfs. Files cached locally from transformers before v3.5.0
    are not shared with those new files, because the cached file's name contains a hash of the url (which changed).
    """
    if subfolder is not None:
        filename = f'{subfolder}/{filename}'
    if mirror:
        endpoint = PRESET_MIRROR_DICT.get(mirror, mirror)
        legacy_format = '/' not in model_id
        if legacy_format:
            return f'{endpoint}/{model_id}-{filename}'
        else:
            return f'{endpoint}/{model_id}/{filename}'
    if revision is None:
        revision = 'main'
    return HUGGINGFACE_CO_PREFIX.format(model_id=model_id, revision=
        revision, filename=filename)

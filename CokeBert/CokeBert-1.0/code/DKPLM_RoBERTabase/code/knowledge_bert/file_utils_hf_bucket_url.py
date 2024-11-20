def hf_bucket_url(identifier, postfix=None, cdn=False):
    endpoint = CLOUDFRONT_DISTRIB_PREFIX if cdn else S3_BUCKET_PREFIX
    if postfix is None:
        return '/'.join((endpoint, identifier))
    else:
        return '/'.join((endpoint, identifier, postfix))

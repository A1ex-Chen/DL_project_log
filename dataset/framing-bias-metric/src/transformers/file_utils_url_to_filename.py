def url_to_filename(url: str, etag: Optional[str]=None) ->str:
    """
    Convert `url` into a hashed filename in a repeatable way. If `etag` is specified, append its hash to the url's,
    delimited by a period. If the url ends with .h5 (Keras HDF5 weights) adds '.h5' to the name so that TF 2.0 can
    identify it as a HDF5 file (see
    https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1380)
    """
    url_bytes = url.encode('utf-8')
    filename = sha256(url_bytes).hexdigest()
    if etag:
        etag_bytes = etag.encode('utf-8')
        filename += '.' + sha256(etag_bytes).hexdigest()
    if url.endswith('.h5'):
        filename += '.h5'
    return filename

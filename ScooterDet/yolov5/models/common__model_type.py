@staticmethod
def _model_type(p='path/to/model.pt'):
    from export import export_formats
    from utils.downloads import is_url
    sf = list(export_formats().Suffix)
    if not is_url(p, check=False):
        check_suffix(p, sf)
    url = urlparse(p)
    types = [(s in Path(p).name) for s in sf]
    types[8] &= not types[9]
    triton = not any(types) and all([any(s in url.scheme for s in ['http',
        'grpc']), url.netloc])
    return types + [triton]

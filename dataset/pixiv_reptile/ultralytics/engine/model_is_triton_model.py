@staticmethod
def is_triton_model(model: str) ->bool:
    """Is model a Triton Server URL string, i.e. <scheme>://<netloc>/<endpoint>/<task_name>"""
    from urllib.parse import urlsplit
    url = urlsplit(model)
    return url.netloc and url.path and url.scheme in {'http', 'grpc'}

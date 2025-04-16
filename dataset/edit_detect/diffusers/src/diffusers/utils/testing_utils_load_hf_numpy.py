def load_hf_numpy(path) ->np.ndarray:
    base_url = (
        'https://huggingface.co/datasets/fusing/diffusers-testing/resolve/main'
        )
    if not path.startswith('http://') and not path.startswith('https://'):
        path = os.path.join(base_url, urllib.parse.quote(path))
    return load_numpy(path)

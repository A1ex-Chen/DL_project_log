def load_hf_numpy(path) ->np.ndarray:
    if not path.startswith('http://') or path.startswith('https://'):
        path = os.path.join(
            'https://huggingface.co/datasets/fusing/diffusers-testing/resolve/main'
            , urllib.parse.quote(path))
    return load_numpy(path)

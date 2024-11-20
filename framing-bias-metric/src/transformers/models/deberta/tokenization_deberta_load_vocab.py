def load_vocab(name=None, tag=None, no_cache=False, cache_dir=None):
    import torch
    if name is None:
        name = 'bpe_encoder'
    model_path = name
    if model_path and not os.path.exists(model_path) and not ('/' in
        model_path or '\\' in model_path):
        _tag = tag
        if _tag is None:
            _tag = 'latest'
        if not cache_dir:
            cache_dir = os.path.join(pathlib.Path.home(),
                f'.~DeBERTa/assets/{_tag}/')
        os.makedirs(cache_dir, exist_ok=True)
        out_dir = os.path.join(cache_dir, name)
        model_path = os.path.join(out_dir, 'bpe_encoder.bin')
        if not os.path.exists(model_path) or no_cache:
            asset = download_asset(name + '.zip', tag=tag, no_cache=
                no_cache, cache_dir=cache_dir)
            with ZipFile(asset, 'r') as zipf:
                for zip_info in zipf.infolist():
                    if zip_info.filename[-1] == '/':
                        continue
                    zip_info.filename = os.path.basename(zip_info.filename)
                    zipf.extract(zip_info, out_dir)
    elif not model_path:
        return None, None
    encoder_state = torch.load(model_path)
    return encoder_state

def download(model_name):
    url = MODELS_MAP[model_name]['url']
    r = requests.get(url, stream=True)
    local_filename = f'./{model_name}.ckpt'
    with open(local_filename, 'wb') as fp:
        for chunk in r.iter_content(chunk_size=8192):
            fp.write(chunk)
    return local_filename

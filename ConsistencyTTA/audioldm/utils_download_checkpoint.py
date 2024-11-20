def download_checkpoint(checkpoint_name='audioldm-s-full'):
    meta = get_metadata()
    if checkpoint_name not in meta.keys():
        print(
            'The model name you provided is not supported. Please use one of the following: '
            , meta.keys())
    if not os.path.exists(meta[checkpoint_name]['path']) or os.path.getsize(
        meta[checkpoint_name]['path']) < 2 * 10 ** 9:
        os.makedirs(os.path.dirname(meta[checkpoint_name]['path']),
            exist_ok=True)
        print(
            f"Downloading the main structure of {checkpoint_name} into {os.path.dirname(meta[checkpoint_name]['path'])}"
            )
        urllib.request.urlretrieve(meta[checkpoint_name]['url'], meta[
            checkpoint_name]['path'], MyProgressBar())
        print('Weights downloaded in: {} Size: {}'.format(meta[
            checkpoint_name]['path'], os.path.getsize(meta[checkpoint_name]
            ['path'])))

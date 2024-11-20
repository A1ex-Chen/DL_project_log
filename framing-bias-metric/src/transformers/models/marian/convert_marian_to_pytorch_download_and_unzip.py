def download_and_unzip(url, dest_dir):
    try:
        import wget
    except ImportError:
        raise ImportError('you must pip install wget')
    filename = wget.download(url)
    unzip(filename, dest_dir)
    os.remove(filename)

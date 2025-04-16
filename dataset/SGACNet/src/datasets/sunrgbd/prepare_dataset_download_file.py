def download_file(url, output_filepath, display_progressbar=False):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=
        url.split('/')[-1], disable=not display_progressbar) as t:
        urllib.request.urlretrieve(url, filename=output_filepath,
            reporthook=t.update_to)

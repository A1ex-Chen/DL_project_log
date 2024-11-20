def download_google_drive_url(url: str, output_path: str, output_file_name: str
    ):
    """
    Download a file from google drive
    Downloading an URL from google drive requires confirmation when
    the file of the size is too big (google drive notifies that
    anti-viral checks cannot be performed on such files)
    """
    import requests
    with requests.Session() as session:
        with session.get(url, stream=True, allow_redirects=True) as response:
            for k, v in response.cookies.items():
                if k.startswith('download_warning'):
                    url = url + '&confirm=' + v
        with session.get(url, stream=True, verify=True) as response:
            makedir(output_path)
            path = os.path.join(output_path, output_file_name)
            total_size = int(response.headers.get('Content-length', 0))
            with open(path, 'wb') as file:
                from tqdm import tqdm
                with tqdm(total=total_size) as progress_bar:
                    for block in response.iter_content(chunk_size=io.
                        DEFAULT_BUFFER_SIZE):
                        file.write(block)
                        progress_bar.update(len(block))

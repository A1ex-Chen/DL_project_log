def get_file_from_modac(fname, origin):
    """Downloads a file from the "Model and Data Clearning House" (MoDAC)
    repository. Users should already have a MoDAC account to download the data.
    Accounts can be created on modac.cancer.gov

        Parameters
        ----------
        fname : string
            path on disk to save the file
        origin : string
            original MoDAC URL of the file

        Returns
        ----------
        string
            Path to the downloaded file
    """
    print(
        'Downloading data from modac.cancer.gov, make sure you have an account first.'
        )
    total_size_in_bytes = get_dataObject_modac_filesize(origin)
    modac_user, modac_token = authenticate_modac()
    data = json.dumps({})
    headers = {}
    headers['Content-Type'] = 'application/json'
    headers['Authorization'] = 'Bearer {0}'.format(modac_token)
    post_url = origin + '/download'
    print('Downloading: ' + post_url + ' ...')
    response = requests.post(post_url, data=data, headers=headers, stream=True)
    if response.status_code != 200:
        print('Error downloading from modac.cancer.gov')
        raise Exception('Response code: {0}, Response message: {1}'.format(
            response.status_code, response.text))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(fname, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        raise Exception('ERROR, something went wrong while downloading ',
            post_url)
    print('Saved file to: ' + fname)
    return fname

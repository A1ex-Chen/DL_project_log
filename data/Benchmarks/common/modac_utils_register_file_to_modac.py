def register_file_to_modac(file_path, metadata, destination_path):
    """Register a file in the "Model and Data Clearning House" (MoDAC).
    The file size is limited to 2GBs

    Parameters
    ----------
    file_path : string
        path on disk for the file to be uploaded
    metadata: dictionary
        dictionary of attribute/value pairs of metadata to associate with
        the file in MoDaC
    destination : string
        The path on MoDaC in form of collection/filename

    Returns
    ----------
    integer
        The returned code from the PUT request
    """
    print('Registering the file {0} at MoDaC location:{1}'.format(file_path,
        destination_path))
    register_url = ('https://modac.cancer.gov/api/v2/dataObject/' +
        destination_path)
    formated_metadata = [dict([('attribute', attribute), ('value', metadata
        [attribute])]) for attribute in metadata.keys()]
    metadata_dict = {'metadataEntries': formated_metadata}
    files = {}
    files['dataObjectRegistration'] = 'attributes', json.dumps(metadata_dict
        ), 'application/json'
    files['dataObject'] = file_path, open(file_path, 'rb')
    modac_user, modac_token = authenticate_modac()
    headers = {}
    headers['Authorization'] = 'Bearer {0}'.format(modac_token)
    response = requests.put(register_url, headers=headers, files=files)
    if response.status_code != 200:
        print(response.headers)
        print(response.text)
        print('Error registering file to modac.cancer.gov')
        raise Exception('Response code: {0}, Response message: {1}'.format(
            response.status_code, response.text))
    print(response.text, response.status_code)
    return response.status_code

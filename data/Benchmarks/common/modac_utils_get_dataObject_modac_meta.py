def get_dataObject_modac_meta(data_object_path):
    """
    Return the self metadata values for a file (data_object)
        Parameters
        ----------
        data_object_path : string
            The path of the file on MoDAC

        Returns
        ----------
        dictionary
            Dictonary of all metadata for the file in MoDAC
    """
    modac_user, modac_token = authenticate_modac()
    headers = {}
    headers['Authorization'] = 'Bearer {0}'.format(modac_token)
    get_response = requests.get(data_object_path, headers=headers)
    if get_response.status_code != 200:
        print('Error downloading from modac.cancer.gov', data_object_path)
        raise Exception('Response code: {0}, Response message: {1}'.format(
            get_response.status_code, get_response.text))
    metadata_dic = json.loads(get_response.text)
    self_metadata = metadata_dic['metadataEntries']['selfMetadataEntries'][
        'systemMetadataEntries']
    self_dic = {}
    for pair in self_metadata:
        self_dic[pair['attribute']] = pair['value']
    return self_dic

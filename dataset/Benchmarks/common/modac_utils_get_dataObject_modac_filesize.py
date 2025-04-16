def get_dataObject_modac_filesize(data_object_path):
    """
    Return the file size in bytes for a modac file
        Parameters
        ----------
        data_object_path : string
            The path of the file on MoDAC

        Returns
        ----------
        integer
            file size in bytes
    """
    self_dic = get_dataObject_modac_meta(data_object_path)
    if 'source_file_size' in self_dic.keys():
        return int(self_dic['source_file_size'])
    else:
        return None

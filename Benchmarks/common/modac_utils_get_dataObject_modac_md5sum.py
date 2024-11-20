def get_dataObject_modac_md5sum(data_object_path):
    """
    Return the md5sum for a modac file
        Parameters
        ----------
        data_object_path : string
            The path of the file on MoDAC

        Returns
        ----------
        string
            The md5sum of the file
    """
    self_dic = get_dataObject_modac_meta(data_object_path)
    if 'checksum' in self_dic.keys():
        return self_dic['checksum']
    else:
        return None

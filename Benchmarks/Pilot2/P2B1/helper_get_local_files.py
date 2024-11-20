def get_local_files(data_tag='3k_run16', data_dir_prefix=
    '/p/gscratchr/brainusr/datasets/cancer/pilot2'):
    """
    Load data files from local directory
    """
    if data_tag == '3k_run16':
        data_dir = (data_dir_prefix +
            '/3k_run16_10us.35fs-DPPC.20-DIPC.60-CHOL.20.dir/')
    elif data_tag == '3k_run10':
        data_dir = (data_dir_prefix +
            '/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/')
    elif data_tag == '3k_run32':
        data_dir = (data_dir_prefix +
            '/3k_run32_10us.35fs-DPPC.50-DOPC.10-CHOL.40.dir/')
    data_files = glob.glob('%s/*.npz' % data_dir)
    filelist = [d for d in data_files if 'AE' not in d]
    filelist = sorted(filelist)
    import pilot2_datasets as p2
    fields = p2.gen_data_set_dict()
    return filelist, fields

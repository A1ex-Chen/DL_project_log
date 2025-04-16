def extract(fpath, dest_folder):
    if fpath.endswith('.tar.gz'):
        mode = 'r:gz'
    elif fpath.endswith('.tar'):
        mode = 'r:'
    else:
        raise IOError('fpath has unknown extention: %s' % fpath)
    with tarfile.open(fpath, mode) as tar:
        members = tar.getmembers()
        for member in tqdm.tqdm(iterable=members, total=len(members), leave
            =True):
            tar.extract(path=dest_folder, member=member)

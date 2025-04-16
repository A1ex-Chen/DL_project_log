def md5_checksum(fpath, target_hash):
    file_hash = hashlib.md5()
    with open(fpath, 'rb') as fp:
        for chunk in iter(lambda : fp.read(1024 * 1024), b''):
            file_hash.update(chunk)
    return file_hash.hexdigest() == target_hash

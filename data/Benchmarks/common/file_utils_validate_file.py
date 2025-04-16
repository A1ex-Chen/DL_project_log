def validate_file(fpath, md5_hash):
    """Validates a file against a MD5 hash

    Parameters
    ----------
    fpath : string
        path to the file being validated
    md5_hash : string
        the MD5 hash being validated against

    Returns
    ----------
    boolean
        Whether the file is valid
    """
    hasher = hashlib.md5()
    with open(fpath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    if str(hasher.hexdigest()) == str(md5_hash):
        return True
    else:
        return False

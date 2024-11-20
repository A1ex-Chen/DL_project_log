def safe_extract(tar, path='.', members=None, *, numeric_owner=False):
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not is_within_directory(path, member_path):
            raise Exception('Attempted Path Traversal in Tar File')
    tar.extractall(path, members, numeric_owner=numeric_owner)

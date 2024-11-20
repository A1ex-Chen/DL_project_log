@staticmethod
def get_hash(paths):
    """Get the hash value of paths"""
    assert isinstance(paths, list), 'Only support list currently.'
    h = hashlib.md5(''.join(paths).encode())
    return h.hexdigest()

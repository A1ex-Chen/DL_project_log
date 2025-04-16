def get_hash(paths):
    """Returns a single hash value of a list of paths (files or dirs)."""
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))
    h = hashlib.sha256(str(size).encode())
    h.update(''.join(paths).encode())
    return h.hexdigest()

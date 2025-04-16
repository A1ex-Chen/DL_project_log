def get_file_hash(filename, max_has_symbols=16, min_large_file_size_mb=1000):
    filesize_mb = os.path.getsize(filename) / (KB_IN_MB_COUNT * KB_IN_MB_COUNT)
    is_large_file = filesize_mb > min_large_file_size_mb
    sha256_hash = hashlib.sha256()
    with open(filename, 'rb') as f:
        if is_large_file:
            for byte_block in iter(lambda : f.read(4096), b''):
                sha256_hash.update(byte_block)
            readable_hash = sha256_hash.hexdigest()
        else:
            bytes = f.read()
            readable_hash = hashlib.sha256(bytes).hexdigest()
    return readable_hash[:max_has_symbols]

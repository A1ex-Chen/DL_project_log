def get_hash(files):
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))

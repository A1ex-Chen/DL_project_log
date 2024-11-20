def build_directory(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path, exist_ok=True)

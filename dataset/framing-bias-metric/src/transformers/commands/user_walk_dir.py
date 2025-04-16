def walk_dir(self, rel_path):
    """
        Recursively list all files in a folder.
        """
    entries: List[os.DirEntry] = list(os.scandir(rel_path))
    files = [(os.path.join(os.getcwd(), f.path), f.path) for f in entries if
        f.is_file()]
    for f in entries:
        if f.is_dir():
            files += self.walk_dir(f.path)
    return files

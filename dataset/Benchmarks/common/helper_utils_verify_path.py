def verify_path(path):
    """Verify if a directory path exists locally. If the path
    does not exist, but is a valid path, it recursivelly creates
    the specified directory path structure.

    Parameters
    ----------
    path : directory path
        Description of local directory path
    """
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

def list_dir(root, prefix=False):
    """List all directories at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(filter(lambda p: os.path.isdir(os.path.join(root, p)
        ), os.listdir(root)))
    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]
    return directories

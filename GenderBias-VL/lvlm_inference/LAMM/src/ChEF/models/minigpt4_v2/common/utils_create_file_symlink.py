def create_file_symlink(file1, file2):
    """
    Simply create the symlinks for a given file1 to file2.
    Useful during model checkpointing to symlinks to the
    latest successful checkpoint.
    """
    try:
        if g_pathmgr.exists(file2):
            g_pathmgr.rm(file2)
        g_pathmgr.symlink(file1, file2)
    except Exception as e:
        logging.info(f'Could NOT create symlink. Error: {e}')

def cleanup_dir(dir):
    """
    Utility for deleting a directory. Useful for cleaning the storage space
    that contains various training artifacts like checkpoints, data etc.
    """
    if os.path.exists(dir):
        logging.info(f'Deleting directory: {dir}')
        shutil.rmtree(dir)
    logging.info(f'Deleted contents of directory: {dir}')

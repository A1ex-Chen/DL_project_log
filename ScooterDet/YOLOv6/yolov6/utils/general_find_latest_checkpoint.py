def find_latest_checkpoint(search_dir='.'):
    """Find the most recent saved checkpoint in search_dir."""
    checkpoint_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(checkpoint_list, key=os.path.getctime
        ) if checkpoint_list else ''

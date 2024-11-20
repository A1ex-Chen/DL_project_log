def get_latest_run(search_dir='.'):
    """Return path to most recent 'last.pt' in /runs (i.e. to --resume from)."""
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''

@contextmanager
def _resumable_file_manager():
    with open(incomplete_path, 'a+b') as f:
        yield f

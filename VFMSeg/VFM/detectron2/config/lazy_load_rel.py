@staticmethod
def load_rel(filename: str, keys: Union[None, str, Tuple[str, ...]]=None):
    """
        Similar to :meth:`load()`, but load path relative to the caller's
        source file.

        This has the same functionality as a relative import, except that this method
        accepts filename as a string, so more characters are allowed in the filename.
        """
    caller_frame = inspect.stack()[1]
    caller_fname = caller_frame[0].f_code.co_filename
    assert caller_fname != '<string>', 'load_rel Unable to find caller'
    caller_dir = os.path.dirname(caller_fname)
    filename = os.path.join(caller_dir, filename)
    return LazyConfig.load(filename, keys)

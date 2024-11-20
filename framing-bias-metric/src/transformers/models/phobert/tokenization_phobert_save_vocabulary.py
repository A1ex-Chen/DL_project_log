def save_vocabulary(self, save_directory: str, filename_prefix: Optional[
    str]=None) ->Tuple[str]:
    if not os.path.isdir(save_directory):
        logger.error('Vocabulary path ({}) should be a directory'.format(
            save_directory))
        return
    out_vocab_file = os.path.join(save_directory, (filename_prefix + '-' if
        filename_prefix else '') + VOCAB_FILES_NAMES['vocab_file'])
    out_merge_file = os.path.join(save_directory, (filename_prefix + '-' if
        filename_prefix else '') + VOCAB_FILES_NAMES['merges_file'])
    if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
        copyfile(self.vocab_file, out_vocab_file)
    if os.path.abspath(self.merges_file) != os.path.abspath(out_merge_file):
        copyfile(self.merges_file, out_merge_file)
    return out_vocab_file, out_merge_file

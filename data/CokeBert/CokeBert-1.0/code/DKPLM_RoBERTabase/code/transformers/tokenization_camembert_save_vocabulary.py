def save_vocabulary(self, save_directory):
    """ Save the sentencepiece vocabulary (copy original file) and special tokens file
            to a directory.
        """
    if not os.path.isdir(save_directory):
        logger.error('Vocabulary path ({}) should be a directory'.format(
            save_directory))
        return
    out_vocab_file = os.path.join(save_directory, VOCAB_FILES_NAMES[
        'vocab_file'])
    if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
        copyfile(self.vocab_file, out_vocab_file)
    return out_vocab_file,

def save_vocabulary(self, save_directory, filename_prefix: Optional[str]=None
    ) ->Tuple[str]:
    """Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
    if not os.path.isdir(save_directory):
        logger.error(
            f'Vocabulary path ({save_directory}) should be a directory')
        return
    out_vocab_file = os.path.join(save_directory, (filename_prefix + '-' if
        filename_prefix else '') + VOCAB_FILES_NAMES['vocab_file'])
    if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file
        ) and os.path.isfile(self.vocab_file):
        copyfile(self.vocab_file, out_vocab_file)
    elif not os.path.isfile(self.vocab_file):
        with open(out_vocab_file, 'wb') as fi:
            content_spiece_model = self.sp_model.serialized_model_proto()
            fi.write(content_spiece_model)
    return out_vocab_file,

def save_vocabulary(self, save_directory: str, filename_prefix: Optional[
    str]=None) ->Tuple[str]:
    if os.path.isdir(save_directory):
        vocab_file = os.path.join(save_directory, (filename_prefix + '-' if
            filename_prefix else '') + VOCAB_FILES_NAMES[
            'pretrained_vocab_file'])
    else:
        vocab_file = (filename_prefix + '-' if filename_prefix else ''
            ) + save_directory
    with open(vocab_file, 'wb') as f:
        pickle.dump(self.__dict__, f)
    return vocab_file,

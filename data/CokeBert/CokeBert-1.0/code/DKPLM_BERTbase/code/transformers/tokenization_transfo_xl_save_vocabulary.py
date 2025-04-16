def save_vocabulary(self, vocab_path):
    """Save the tokenizer vocabulary to a directory or file."""
    if os.path.isdir(vocab_path):
        vocab_file = os.path.join(vocab_path, VOCAB_FILES_NAMES[
            'pretrained_vocab_file'])
    torch.save(self.__dict__, vocab_file)
    return vocab_file,

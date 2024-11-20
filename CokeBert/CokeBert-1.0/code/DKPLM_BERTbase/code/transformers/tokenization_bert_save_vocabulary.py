def save_vocabulary(self, vocab_path):
    """Save the tokenizer vocabulary to a directory or file."""
    index = 0
    if os.path.isdir(vocab_path):
        vocab_file = os.path.join(vocab_path, VOCAB_FILES_NAMES['vocab_file'])
    else:
        vocab_file = vocab_path
    with open(vocab_file, 'w', encoding='utf-8') as writer:
        for token, token_index in sorted(self.vocab.items(), key=lambda kv:
            kv[1]):
            if index != token_index:
                logger.warning(
                    'Saving vocabulary to {}: vocabulary indices are not consecutive. Please check that the vocabulary is not corrupted!'
                    .format(vocab_file))
                index = token_index
            writer.write(token + u'\n')
            index += 1
    return vocab_file,

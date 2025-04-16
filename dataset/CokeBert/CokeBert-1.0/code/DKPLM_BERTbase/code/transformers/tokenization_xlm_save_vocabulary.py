def save_vocabulary(self, save_directory):
    """Save the tokenizer vocabulary and merge files to a directory."""
    if not os.path.isdir(save_directory):
        logger.error('Vocabulary path ({}) should be a directory'.format(
            save_directory))
        return
    vocab_file = os.path.join(save_directory, VOCAB_FILES_NAMES['vocab_file'])
    merge_file = os.path.join(save_directory, VOCAB_FILES_NAMES['merges_file'])
    with open(vocab_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(self.encoder, ensure_ascii=False))
    index = 0
    with open(merge_file, 'w', encoding='utf-8') as writer:
        for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=
            lambda kv: kv[1]):
            if index != token_index:
                logger.warning(
                    'Saving vocabulary to {}: BPE merge indices are not consecutive. Please check that the tokenizer is not corrupted!'
                    .format(merge_file))
                index = token_index
            writer.write(' '.join(bpe_tokens) + u'\n')
            index += 1
    return vocab_file, merge_file

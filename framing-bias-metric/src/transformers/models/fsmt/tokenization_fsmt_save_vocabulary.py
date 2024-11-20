def save_vocabulary(self, save_directory: str, filename_prefix: Optional[
    str]=None) ->Tuple[str]:
    if not os.path.isdir(save_directory):
        logger.error('Vocabulary path ({}) should be a directory'.format(
            save_directory))
        return
    src_vocab_file = os.path.join(save_directory, (filename_prefix + '-' if
        filename_prefix else '') + VOCAB_FILES_NAMES['src_vocab_file'])
    tgt_vocab_file = os.path.join(save_directory, (filename_prefix + '-' if
        filename_prefix else '') + VOCAB_FILES_NAMES['tgt_vocab_file'])
    merges_file = os.path.join(save_directory, (filename_prefix + '-' if
        filename_prefix else '') + VOCAB_FILES_NAMES['merges_file'])
    with open(src_vocab_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(self.encoder, ensure_ascii=False))
    with open(tgt_vocab_file, 'w', encoding='utf-8') as f:
        tgt_vocab = {v: k for k, v in self.decoder.items()}
        f.write(json.dumps(tgt_vocab, ensure_ascii=False))
    index = 0
    with open(merges_file, 'w', encoding='utf-8') as writer:
        for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=
            lambda kv: kv[1]):
            if index != token_index:
                logger.warning(
                    'Saving vocabulary to {}: BPE merge indices are not consecutive. Please check that the tokenizer is not corrupted!'
                    .format(merges_file))
                index = token_index
            writer.write(' '.join(bpe_tokens) + '\n')
            index += 1
    return src_vocab_file, tgt_vocab_file, merges_file

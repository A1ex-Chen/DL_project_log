def _save_pretrained(self, save_directory: Union[str, os.PathLike],
    file_names: Tuple[str], legacy_format: bool=True, filename_prefix:
    Optional[str]=None) ->Tuple[str]:
    """
        Save a tokenizer using the slow-tokenizer/legacy format: vocabulary + added tokens.

        Fast tokenizers can also be saved in a unique JSON file containing {config + vocab + added-tokens} using the
        specific :meth:`~transformers.PreTrainedTokenizerFast._save_pretrained`
        """
    save_directory = str(save_directory)
    if legacy_format:
        added_tokens_file = os.path.join(save_directory, (filename_prefix +
            '-' if filename_prefix else '') + ADDED_TOKENS_FILE)
        added_vocab = self.get_added_vocab()
        if added_vocab:
            with open(added_tokens_file, 'w', encoding='utf-8') as f:
                out_str = json.dumps(added_vocab, ensure_ascii=False)
                f.write(out_str)
        vocab_files = self.save_vocabulary(save_directory, filename_prefix=
            filename_prefix)
        file_names = file_names + vocab_files + (added_tokens_file,)
    else:
        tokenizer_file = os.path.join(save_directory, (filename_prefix +
            '-' if filename_prefix else '') + TOKENIZER_FILE)
        self.backend_tokenizer.save(tokenizer_file)
        file_names = file_names + (tokenizer_file,)
    return file_names

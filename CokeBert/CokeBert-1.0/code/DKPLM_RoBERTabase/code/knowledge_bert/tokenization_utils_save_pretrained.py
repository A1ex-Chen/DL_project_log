def save_pretrained(self, save_directory):
    """ Save the tokenizer vocabulary files together with:
                - added tokens,
                - special-tokens-to-class-attributes-mapping,
                - tokenizer instantiation positional and keywords inputs (e.g. do_lower_case for Bert).

            This won't save modifications other than (added tokens and special token mapping) you may have
            applied to the tokenizer after the instantiation (e.g. modifying tokenizer.do_lower_case after creation).

            This method make sure the full tokenizer can then be re-loaded using the :func:`~transformers.PreTrainedTokenizer.from_pretrained` class method.
        """
    if not os.path.isdir(save_directory):
        logger.error('Saving directory ({}) should be a directory'.format(
            save_directory))
        return
    special_tokens_map_file = os.path.join(save_directory,
        SPECIAL_TOKENS_MAP_FILE)
    added_tokens_file = os.path.join(save_directory, ADDED_TOKENS_FILE)
    tokenizer_config_file = os.path.join(save_directory, TOKENIZER_CONFIG_FILE)
    tokenizer_config = copy.deepcopy(self.init_kwargs)
    tokenizer_config['init_inputs'] = copy.deepcopy(self.init_inputs)
    for file_id in self.vocab_files_names.keys():
        tokenizer_config.pop(file_id, None)
    with open(tokenizer_config_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_config, ensure_ascii=False))
    with open(special_tokens_map_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(self.special_tokens_map, ensure_ascii=False))
    with open(added_tokens_file, 'w', encoding='utf-8') as f:
        if self.added_tokens_encoder:
            out_str = json.dumps(self.added_tokens_encoder, ensure_ascii=False)
        else:
            out_str = u'{}'
        f.write(out_str)
    vocab_files = self.save_vocabulary(save_directory)
    return vocab_files + (special_tokens_map_file, added_tokens_file)

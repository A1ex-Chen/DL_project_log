def save_pretrained(self, save_directory: Union[str, os.PathLike],
    legacy_format: bool=True, filename_prefix: Optional[str]=None) ->Tuple[str
    ]:
    """
        Save the full tokenizer state.


        This method make sure the full tokenizer can then be re-loaded using the
        :meth:`~transformers.tokenization_utils_base.PreTrainedTokenizer.from_pretrained` class method.

        .. Note::
            A "fast" tokenizer (instance of :class:`transformers.PreTrainedTokenizerFast`) saved with this method will
            not be possible to load back in a "slow" tokenizer, i.e. in a :class:`transformers.PreTrainedTokenizer`
            instance. It can only be loaded in a "fast" tokenizer, i.e. in a
            :class:`transformers.PreTrainedTokenizerFast` instance.

        .. Warning::
           This won't save modifications you may have applied to the tokenizer after the instantiation (for instance,
           modifying :obj:`tokenizer.do_lower_case` after creation).

        Args:
            save_directory (:obj:`str` or :obj:`os.PathLike`): The path to a directory where the tokenizer will be saved.
            legacy_format (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether to save the tokenizer in legacy format (default), i.e. with tokenizer specific vocabulary and a
                separate added_tokens files or in the unified JSON file format for the `tokenizers` library. It's only
                possible to save a Fast tokenizer in the unified JSON format and this format is incompatible with
                "slow" tokenizers (not powered by the `tokenizers` library).
            filename_prefix: (:obj:`str`, `optional`):
                A prefix to add to the names of the files saved by the tokenizer.

        Returns:
            A tuple of :obj:`str`: The files saved.
        """
    if os.path.isfile(save_directory):
        logger.error('Provided path ({}) should be a directory, not a file'
            .format(save_directory))
        return
    os.makedirs(save_directory, exist_ok=True)
    special_tokens_map_file = os.path.join(save_directory, (filename_prefix +
        '-' if filename_prefix else '') + SPECIAL_TOKENS_MAP_FILE)
    tokenizer_config_file = os.path.join(save_directory, (filename_prefix +
        '-' if filename_prefix else '') + TOKENIZER_CONFIG_FILE)
    tokenizer_config = copy.deepcopy(self.init_kwargs)
    if len(self.init_inputs) > 0:
        tokenizer_config['init_inputs'] = copy.deepcopy(self.init_inputs)
    for file_id in self.vocab_files_names.keys():
        tokenizer_config.pop(file_id, None)

    def convert_added_tokens(obj: Union[AddedToken, Any], add_type_field=True):
        if isinstance(obj, AddedToken):
            out = obj.__getstate__()
            if add_type_field:
                out['__type'] = 'AddedToken'
            return out
        elif isinstance(obj, (list, tuple)):
            return list(convert_added_tokens(o, add_type_field=
                add_type_field) for o in obj)
        elif isinstance(obj, dict):
            return {k: convert_added_tokens(v, add_type_field=
                add_type_field) for k, v in obj.items()}
        return obj
    tokenizer_config = convert_added_tokens(tokenizer_config,
        add_type_field=True)
    with open(tokenizer_config_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_config, ensure_ascii=False))
    write_dict = convert_added_tokens(self.special_tokens_map_extended,
        add_type_field=False)
    with open(special_tokens_map_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(write_dict, ensure_ascii=False))
    file_names = tokenizer_config_file, special_tokens_map_file
    return self._save_pretrained(save_directory=save_directory, file_names=
        file_names, legacy_format=legacy_format, filename_prefix=
        filename_prefix)

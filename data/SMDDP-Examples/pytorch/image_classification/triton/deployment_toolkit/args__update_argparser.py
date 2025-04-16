def _update_argparser(self, parser):
    label = 'argparser_update'
    if self._input_path:
        update_argparser_handle = load_from_file(self._input_path, label=
            label, target=GET_ARGPARSER_FN_NAME)
        if update_argparser_handle:
            update_argparser_handle(parser)
        elif self._required_fn_name_for_signature_parsing:
            fn_handle = load_from_file(self._input_path, label=label,
                target=self._required_fn_name_for_signature_parsing)
            if fn_handle:
                add_args_for_fn_signature(parser, fn_handle)

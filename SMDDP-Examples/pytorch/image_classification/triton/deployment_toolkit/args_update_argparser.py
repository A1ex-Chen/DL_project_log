def update_argparser(self, parser):
    name = self._handle.__name__
    group_parser = parser.add_argument_group(name)
    add_args_for_fn_signature(group_parser, fn=self._handle)
    self._update_argparser(group_parser)

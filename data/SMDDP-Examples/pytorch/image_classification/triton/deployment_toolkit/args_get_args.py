def get_args(self, args: argparse.Namespace):
    filtered_args = filter_fn_args(args, fn=self._handle)
    tmp_parser = argparse.ArgumentParser(allow_abbrev=False)
    self._update_argparser(tmp_parser)
    custom_names = [p.dest.replace('-', '_') for p in tmp_parser._actions if
        not isinstance(p, argparse._HelpAction)]
    custom_params = {n: getattr(args, n) for n in custom_names}
    filtered_args = {**filtered_args, **custom_params}
    return filtered_args

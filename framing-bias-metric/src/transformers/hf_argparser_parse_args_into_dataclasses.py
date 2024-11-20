def parse_args_into_dataclasses(self, args=None, return_remaining_strings=
    False, look_for_args_file=True, args_filename=None) ->Tuple[DataClass, ...
    ]:
    """
        Parse command-line args into instances of the specified dataclass types.

        This relies on argparse's `ArgumentParser.parse_known_args`. See the doc at:
        docs.python.org/3.7/library/argparse.html#argparse.ArgumentParser.parse_args

        Args:
            args:
                List of strings to parse. The default is taken from sys.argv. (same as argparse.ArgumentParser)
            return_remaining_strings:
                If true, also return a list of remaining argument strings.
            look_for_args_file:
                If true, will look for a ".args" file with the same base name as the entry point script for this
                process, and will append its potential content to the command line args.
            args_filename:
                If not None, will uses this file instead of the ".args" file specified in the previous argument.

        Returns:
            Tuple consisting of:

                - the dataclass instances in the same order as they were passed to the initializer.abspath
                - if applicable, an additional namespace for more (non-dataclass backed) arguments added to the parser
                  after initialization.
                - The potential list of remaining argument strings. (same as argparse.ArgumentParser.parse_known_args)
        """
    if args_filename or look_for_args_file and len(sys.argv):
        if args_filename:
            args_file = Path(args_filename)
        else:
            args_file = Path(sys.argv[0]).with_suffix('.args')
        if args_file.exists():
            fargs = args_file.read_text().split()
            args = fargs + args if args is not None else fargs + sys.argv[1:]
    namespace, remaining_args = self.parse_known_args(args=args)
    outputs = []
    for dtype in self.dataclass_types:
        keys = {f.name for f in dataclasses.fields(dtype)}
        inputs = {k: v for k, v in vars(namespace).items() if k in keys}
        for k in keys:
            delattr(namespace, k)
        obj = dtype(**inputs)
        outputs.append(obj)
    if len(namespace.__dict__) > 0:
        outputs.append(namespace)
    if return_remaining_strings:
        return *outputs, remaining_args
    else:
        if remaining_args:
            raise ValueError(
                f'Some specified arguments are not used by the HfArgumentParser: {remaining_args}'
                )
        return *outputs,

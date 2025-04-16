def parse_args(self, args):
    if 'hints_file' not in args:
        args.hints_file = None
    self.initialize_hints_config(args.hints_file)
    if 'warm_up' in args and args.warm_up is not None:
        self.warm_up = args.warm_up
    if 'measure_for' in args and args.measure_for is not None:
        self.measure_for = args.measure_for

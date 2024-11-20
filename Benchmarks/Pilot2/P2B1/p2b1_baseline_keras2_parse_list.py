def parse_list(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))

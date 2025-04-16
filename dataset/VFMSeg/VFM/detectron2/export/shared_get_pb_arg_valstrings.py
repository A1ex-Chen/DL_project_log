def get_pb_arg_valstrings(pb, arg_name, default_val):
    arg = get_pb_arg(pb, arg_name)
    return list(arg.strings) if arg is not None else default_val

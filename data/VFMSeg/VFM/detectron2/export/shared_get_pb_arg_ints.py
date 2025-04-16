def get_pb_arg_ints(pb, arg_name, default_val):
    arg = get_pb_arg(pb, arg_name)
    return list(map(int, arg.ints)) if arg is not None else default_val

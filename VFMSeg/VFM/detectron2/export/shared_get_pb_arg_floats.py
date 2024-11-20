def get_pb_arg_floats(pb, arg_name, default_val):
    arg = get_pb_arg(pb, arg_name)
    return list(map(float, arg.floats)) if arg is not None else default_val

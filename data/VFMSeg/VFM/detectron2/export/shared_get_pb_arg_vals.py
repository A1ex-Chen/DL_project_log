def get_pb_arg_vals(pb, arg_name, default_val):
    arg = get_pb_arg(pb, arg_name)
    return arg.s if arg is not None else default_val

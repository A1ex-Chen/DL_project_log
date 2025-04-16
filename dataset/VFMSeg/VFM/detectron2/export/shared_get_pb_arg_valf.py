def get_pb_arg_valf(pb, arg_name, default_val):
    arg = get_pb_arg(pb, arg_name)
    return arg.f if arg is not None else default_val

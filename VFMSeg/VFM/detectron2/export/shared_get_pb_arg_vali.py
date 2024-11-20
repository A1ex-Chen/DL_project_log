def get_pb_arg_vali(pb, arg_name, default_val):
    arg = get_pb_arg(pb, arg_name)
    return arg.i if arg is not None else default_val

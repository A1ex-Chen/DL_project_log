def check_set_pb_arg(pb, arg_name, arg_attr, arg_value, allow_override=False):
    arg = get_pb_arg(pb, arg_name)
    if arg is None:
        arg = putils.MakeArgument(arg_name, arg_value)
        assert hasattr(arg, arg_attr)
        pb.arg.extend([arg])
    if allow_override and getattr(arg, arg_attr) != arg_value:
        logger.warning('Override argument {}: {} -> {}'.format(arg_name,
            getattr(arg, arg_attr), arg_value))
        setattr(arg, arg_attr, arg_value)
    else:
        assert arg is not None
        assert getattr(arg, arg_attr
            ) == arg_value, 'Existing value {}, new value {}'.format(getattr
            (arg, arg_attr), arg_value)

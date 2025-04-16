def get_pb_arg(pb, arg_name):
    for x in pb.arg:
        if x.name == arg_name:
            return x
    return None

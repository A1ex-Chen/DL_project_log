def is_using_hvd():
    return hvd.size() > 1

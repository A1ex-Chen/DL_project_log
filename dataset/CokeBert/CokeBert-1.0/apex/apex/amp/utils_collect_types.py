def collect_types(x, types):
    if is_nested(x):
        for y in x:
            collect_types(y, types)
    else:
        types.add(type_string(x))

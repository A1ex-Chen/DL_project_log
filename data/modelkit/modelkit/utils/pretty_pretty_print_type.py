def pretty_print_type(typ):
    typ_str = str(typ)
    if "'" in typ_str:
        typ_str = typ_str.split("'")[1]
    return escape(typ_str)

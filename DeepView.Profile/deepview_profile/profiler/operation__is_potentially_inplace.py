def _is_potentially_inplace(fn_name):
    return fn_name in POTENTIALLY_INPLACE_FUNCTIONS or len(fn_name
        ) > 1 and fn_name[-1] == '_' and fn_name[-2] != '_'

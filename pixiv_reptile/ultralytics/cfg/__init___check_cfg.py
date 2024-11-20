def check_cfg(cfg, hard=True):
    """Validate Ultralytics configuration argument types and values, converting them if necessary."""
    for k, v in cfg.items():
        if v is not None:
            if k in CFG_FLOAT_KEYS and not isinstance(v, (int, float)):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                        )
                cfg[k] = float(v)
            elif k in CFG_FRACTION_KEYS:
                if not isinstance(v, (int, float)):
                    if hard:
                        raise TypeError(
                            f"'{k}={v}' is of invalid type {type(v).__name__}. Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                            )
                    cfg[k] = v = float(v)
                if not 0.0 <= v <= 1.0:
                    raise ValueError(
                        f"'{k}={v}' is an invalid value. Valid '{k}' values are between 0.0 and 1.0."
                        )
            elif k in CFG_INT_KEYS and not isinstance(v, int):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. '{k}' must be an int (i.e. '{k}=8')"
                        )
                cfg[k] = int(v)
            elif k in CFG_BOOL_KEYS and not isinstance(v, bool):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. '{k}' must be a bool (i.e. '{k}=True' or '{k}=False')"
                        )
                cfg[k] = bool(v)

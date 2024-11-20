def cast_marian_config(raw_cfg: Dict[str, str]) ->Dict:
    return {k: _cast_yaml_str(v) for k, v in raw_cfg.items()}

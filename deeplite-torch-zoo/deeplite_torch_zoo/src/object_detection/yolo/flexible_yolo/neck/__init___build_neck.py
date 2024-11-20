def build_neck(neck_name, **kwargs):
    if neck_name not in NECK_MAP:
        raise ValueError(
            f'Neck {neck_name} not supported. Supported neck types: {NECK_MAP.keys()}'
            )
    neck = NECK_MAP[neck_name](**kwargs)
    return neck

def Dataclass2Dict(data):
    data_dir: List[str] = dir(data)
    res: dict = {}
    for field in data_dir:
        if field[:2] != '__' and field[-2:] != '__':
            res[field] = getattr(data, field)
    return res

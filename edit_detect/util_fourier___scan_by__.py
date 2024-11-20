def __scan_by__(self, var: str, conds: Union[int, List[int], Set[int], str,
    List[str], Set[str], None]=None):
    if conds is None or isinstance(conds, list) and len(conds
        ) == 0 or isinstance(conds, set) and len(conds) == 0:
        return True
    elif isinstance(conds, int) or isinstance(conds, str):
        conds = [conds]
    for cond in conds:
        if f'_{cond}' in var:
            return True
    return False

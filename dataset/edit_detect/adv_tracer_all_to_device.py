@staticmethod
def all_to_device(*args, device: Union[str, torch.device]):
    comps: List = []
    for comp in args:
        if hasattr(comp, 'to'):
            comps.append(comp.to(device))
        else:
            comps.append(comp)
    return comps

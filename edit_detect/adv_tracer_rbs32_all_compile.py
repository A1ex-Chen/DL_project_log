@staticmethod
def all_compile(*args):
    comps: List = []
    for comp in args:
        if comp is None:
            comps.append(comp)
        else:
            comps.append(torch.compile(comp, mode='reduce-overhead',
                fullgraph=True))
    return comps

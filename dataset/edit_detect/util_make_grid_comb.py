@staticmethod
def make_grid_comb(models: Union[str, List[str]], datasets: Union[str, List
    [str]]) ->Dict[str, List[str]]:
    if isinstance(models, str):
        models = [models]
    if isinstance(datasets, str):
        datasets = [datasets]
    comb: Dict[str, List[str]] = {}
    for model in models:
        comb[model] = datasets
    return comb

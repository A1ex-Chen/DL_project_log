def __init__(self, datasets: List[wds.DataPipeline]) ->None:
    super().__init__()
    self.datasets = datasets
    self.prob = []
    self.names = []
    for dataset in self.datasets:
        if hasattr(dataset, 'name'):
            self.names.append(dataset.name)
        else:
            self.names.append('Unknown')
        if hasattr(dataset, 'sample_ratio'):
            self.prob.append(dataset.sample_ratio)
        else:
            self.prob.append(1)
            logging.info(
                "One of the datapipeline doesn't define ratio and set to 1 automatically."
                )

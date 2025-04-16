def __iter__(self):
    datastreams = [iter(dataset) for dataset in self.datasets]
    while True:
        select_datastream = random.choices(datastreams, weights=self.prob, k=1
            )[0]
        yield next(select_datastream)

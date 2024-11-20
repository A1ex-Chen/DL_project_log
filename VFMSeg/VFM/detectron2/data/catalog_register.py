def register(self, name, func):
    """
        Args:
            name (str): the name that identifies a dataset, e.g. "coco_2014_train".
            func (callable): a callable which takes no arguments and returns a list of dicts.
                It must return the same results if called multiple times.
        """
    assert callable(func
        ), 'You must register a function with `DatasetCatalog.register`!'
    assert name not in self, "Dataset '{}' is already registered!".format(name)
    self[name] = func

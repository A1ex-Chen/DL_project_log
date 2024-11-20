def __init__(self, dataclass_types: Union[DataClassType, Iterable[
    DataClassType]], **kwargs):
    """
        Args:
            dataclass_types:
                Dataclass type, or list of dataclass types for which we will "fill" instances with the parsed args.
            kwargs:
                (Optional) Passed to `argparse.ArgumentParser()` in the regular way.
        """
    super().__init__(**kwargs)
    if dataclasses.is_dataclass(dataclass_types):
        dataclass_types = [dataclass_types]
    self.dataclass_types = dataclass_types
    for dtype in self.dataclass_types:
        self._add_dataclass_arguments(dtype)

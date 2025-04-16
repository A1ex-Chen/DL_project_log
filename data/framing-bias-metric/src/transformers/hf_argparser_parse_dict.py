def parse_dict(self, args: dict) ->Tuple[DataClass, ...]:
    """
        Alternative helper method that does not use `argparse` at all, instead uses a dict and populating the dataclass
        types.
        """
    outputs = []
    for dtype in self.dataclass_types:
        keys = {f.name for f in dataclasses.fields(dtype)}
        inputs = {k: v for k, v in args.items() if k in keys}
        obj = dtype(**inputs)
        outputs.append(obj)
    return *outputs,

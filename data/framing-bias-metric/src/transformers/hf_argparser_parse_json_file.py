def parse_json_file(self, json_file: str) ->Tuple[DataClass, ...]:
    """
        Alternative helper method that does not use `argparse` at all, instead loading a json file and populating the
        dataclass types.
        """
    data = json.loads(Path(json_file).read_text())
    outputs = []
    for dtype in self.dataclass_types:
        keys = {f.name for f in dataclasses.fields(dtype)}
        inputs = {k: v for k, v in data.items() if k in keys}
        obj = dtype(**inputs)
        outputs.append(obj)
    return *outputs,

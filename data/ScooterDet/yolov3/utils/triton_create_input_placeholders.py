def create_input_placeholders() ->typing.List[InferInput]:
    return [InferInput(i['name'], [int(s) for s in i['shape']], i[
        'datatype']) for i in self.metadata['inputs']]

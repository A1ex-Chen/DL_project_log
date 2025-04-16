def batch_size_location(self):
    """
        Locates the line of the 'batch_size' argument in the
        'skyline_input_provider' function and determines if the provider's
        definition can be mutated using our heuristics.
        """
    extractor = _InputProviderExtractor()
    extractor.visit(self._ast)
    function = extractor.function_node
    if function is None or len(function.args.args) == 0 or function.args.args[0
        ].arg != 'batch_size':
        return None
    batch_size_line_number = function.args.args[0].lineno
    match = END_OF_FUNCTION.search(self._code_by_line[function.lineno - 1])
    can_mutate = match is not None
    return batch_size_line_number, can_mutate

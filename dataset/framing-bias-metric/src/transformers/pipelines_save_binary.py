def save_binary(self, data: Union[dict, List[dict]]) ->str:
    if self.output_path is None:
        raise KeyError(
            'When using piped input on pipeline outputting large object requires an output file path. Please provide such output path through --output argument.'
            )
    return super().save_binary(data)

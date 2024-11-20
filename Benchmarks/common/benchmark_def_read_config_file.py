def read_config_file(self, file):
    """Functionality to read the configue file
        specific for each benchmark.
        """
    config = configparser.ConfigParser()
    config.read(file)
    section = config.sections()
    fileParams = {}
    for sec in section:
        for k, v in config.items(sec):
            if k not in fileParams:
                fileParams[k] = eval(v)
    fileParams = self.format_benchmark_config_arguments(fileParams)
    return fileParams

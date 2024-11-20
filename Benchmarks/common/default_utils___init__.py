def __init__(self, filepath, defmodel, framework, prog=None, desc=None,
    parser=None):
    """Initialize Benchmark object.

        Parameters
        ----------
        filepath : ./
            os.path.dirname where the benchmark is located. Necessary to locate utils and
            establish input/ouput paths
        defmodel : 'p*b*_default_model.txt'
            string corresponding to the default model of the benchmark
        framework : 'keras', 'neon', 'mxnet', 'pytorch'
            framework used to run the benchmark
        prog : 'p*b*_baseline_*'
            string for program name (usually associated to benchmark and framework)
        desc : ' '
            string describing benchmark (usually a description of the neural network model built)
        parser : argparser (default None)
            if 'neon' framework a NeonArgparser is passed. Otherwise an argparser is constructed.
        """
    if parser is None:
        parser = argparse.ArgumentParser(prog=prog, formatter_class=
            argparse.ArgumentDefaultsHelpFormatter, description=desc,
            conflict_handler='resolve')
    self.parser = parser
    self.file_path = filepath
    self.default_model = defmodel
    self.framework = framework
    self.required = set([])
    self.set_locals()

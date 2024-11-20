def format_benchmark_config_arguments(self, dictfileparam):
    """Functionality to format the particular parameters of
        the benchmark.

        Parameters
        ----------
        dictfileparam : python dictionary
            parameters read from configuration file
        args : python dictionary
            parameters read from command-line
            Most of the time command-line overwrites configuration file
            except when the command-line is using default values and
            config file defines those values

        """
    configOut = dictfileparam.copy()
    for d in self.additional_definitions:
        if d['name'] in configOut.keys():
            if 'type' in d:
                dtype = d['type']
            else:
                dtype = None
            if 'action' in d:
                if inspect.isclass(d['action']):
                    str_read = dictfileparam[d['name']]
                    configOut[d['name']] = eval_string_as_list_of_lists(
                        str_read, ':', ',', dtype)
            elif d['default'] != argparse.SUPPRESS:
                self.parser.add_argument('--' + d['name'], type=d['type'],
                    default=configOut[d['name']], help=d['help'])
    return configOut

def parse_from_benchmark(self):
    """Functionality to parse options specific
        specific for each benchmark.
        """
    for d in self.additional_definitions:
        if 'type' not in d:
            d['type'] = None
        if 'default' not in d:
            d['default'] = argparse.SUPPRESS
        if 'help' not in d:
            d['help'] = ''
        if 'action' in d:
            if d['action'] == 'list-of-lists':
                d['action'] = ListOfListsAction
                self.parser.add_argument('--' + d['name'], dest=d['name'],
                    action=d['action'], type=d['type'], default=d['default'
                    ], help=d['help'])
            elif d['action'] == 'store_true' or d['action'] == 'store_false':
                raise Exception(
                    'The usage of store_true or store_false cannot be undone in the command line. Use type=str2bool instead.'
                    )
            else:
                self.parser.add_argument('--' + d['name'], action=d[
                    'action'], default=d['default'], help=d['help'], type=d
                    ['type'])
        elif 'nargs' in d:
            if 'choices' in d:
                self.parser.add_argument('--' + d['name'], nargs=d['nargs'],
                    choices=d['choices'], default=d['default'], help=d['help'])
            else:
                self.parser.add_argument('--' + d['name'], nargs=d['nargs'],
                    type=d['type'], default=d['default'], help=d['help'])
        elif 'choices' in d:
            self.parser.add_argument('--' + d['name'], choices=d['choices'],
                default=d['default'], help=d['help'])
        else:
            self.parser.add_argument('--' + d['name'], type=d['type'],
                default=d['default'], help=d['help'])

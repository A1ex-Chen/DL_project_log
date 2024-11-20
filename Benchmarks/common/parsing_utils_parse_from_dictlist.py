def parse_from_dictlist(dictlist, parser):
    """Functionality to parse options.
    Parameters
    ----------
    pardict : python list of dictionaries
        Specification of parameters
    parser : ArgumentParser object
        Current parser
    """
    for d in dictlist:
        if 'type' not in d:
            d['type'] = None
        if 'default' not in d:
            d['default'] = argparse.SUPPRESS
        if 'help' not in d:
            d['help'] = ''
        if 'abv' not in d:
            d['abv'] = None
        if 'action' in d:
            if d['action'] == 'list-of-lists':
                d['action'] = ListOfListsAction
                if d['abv'] is None:
                    parser.add_argument('--' + d['name'], dest=d['name'],
                        action=d['action'], type=d['type'], default=d[
                        'default'], help=d['help'])
                else:
                    parser.add_argument('-' + d['abv'], '--' + d['name'],
                        dest=d['name'], action=d['action'], type=d['type'],
                        default=d['default'], help=d['help'])
            elif d['action'] == 'store_true' or d['action'] == 'store_false':
                raise Exception(
                    'The usage of store_true or store_false cannot be undone in the command line. Use type=str2bool instead.'
                    )
            elif d['abv'] is None:
                parser.add_argument('--' + d['name'], action=d['action'],
                    default=d['default'], help=d['help'], type=d['type'])
            else:
                parser.add_argument('-' + d['abv'], '--' + d['name'],
                    action=d['action'], default=d['default'], help=d['help'
                    ], type=d['type'])
        elif 'nargs' in d:
            if 'choices' in d:
                if d['abv'] is None:
                    parser.add_argument('--' + d['name'], nargs=d['nargs'],
                        choices=d['choices'], default=d['default'], help=d[
                        'help'])
                else:
                    parser.add_argument('-' + d['abv'], '--' + d['name'],
                        nargs=d['nargs'], choices=d['choices'], default=d[
                        'default'], help=d['help'])
            elif d['abv'] is None:
                parser.add_argument('--' + d['name'], nargs=d['nargs'],
                    type=d['type'], default=d['default'], help=d['help'])
            else:
                parser.add_argument('-' + d['abv'], '--' + d['name'], nargs
                    =d['nargs'], type=d['type'], default=d['default'], help
                    =d['help'])
        elif 'choices' in d:
            if d['abv'] is None:
                parser.add_argument('--' + d['name'], choices=d['choices'],
                    default=d['default'], help=d['help'])
            else:
                parser.add_argument('-' + d['abv'], '--' + d['name'],
                    choices=d['choices'], default=d['default'], help=d['help'])
        elif d['abv'] is None:
            parser.add_argument('--' + d['name'], type=d['type'], default=d
                ['default'], help=d['help'])
        else:
            parser.add_argument('-' + d['abv'], '--' + d['name'], type=d[
                'type'], default=d['default'], help=d['help'])
    return parser

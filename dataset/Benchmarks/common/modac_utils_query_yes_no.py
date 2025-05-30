def query_yes_no(question, default='yes'):
    """
    Ask a yes/no question via raw_input() and return their answer.

        Parameters
        ----------
        question: string
            string that is presented to the user.
        default: boolean
            The presumed boolean answer if the user just hits <Enter>.

        Returns
        ----------
        boolean
            True for "yes" or False for "no".

    """
    valid = {'yes': True, 'y': True, 'ye': True, 'no': False, 'n': False}
    if default is None:
        prompt = ' [y/n] '
    elif default == 'yes':
        prompt = ' [Y/n] '
    elif default == 'no':
        prompt = ' [y/N] '
    else:
        raise ValueError("invalid default answer: '%s'" % default)
    while True:
        print(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")

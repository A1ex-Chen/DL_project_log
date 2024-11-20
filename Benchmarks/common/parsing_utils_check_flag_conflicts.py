def check_flag_conflicts(params):
    """Check if parameters that must be exclusive are used in conjunction.
     The check is made against CONFLICT_LIST, a global list that
     describes parameter pairs that should be exclusive.
     Raises an exception if pairs of parameters in CONFLICT_LIST are
     specified simulataneously.

    Parameters
    ----------
    params : python dictionary
       list to extract keywords from
    """
    key_set = set(params.keys())
    for flag_list in CONFLICT_LIST:
        flag_count = 0
        for i in flag_list:
            if i in key_set:
                if params[i] is True:
                    flag_count += 1
        if flag_count > 1:
            raise Exception(
                'ERROR ! Conflict in flag specification. These flags should not be used together: '
                 + str(sorted(flag_list)) + '... Exiting')

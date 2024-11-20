def check_flag_conflicts(params):
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

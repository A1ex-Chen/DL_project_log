def what_got_skipped(which_iter, which_backward, which_model):
    if which_iter == 0:
        if which_backward == 0:
            if which_model == 0:
                return 1
            if which_model == 1:
                return 2
        if which_backward == 1:
            if which_model == 2:
                return 3
            if which_model == 1:
                return 4
    if which_iter == 1:
        if which_backward == 0:
            if which_model == 0:
                return 5
            if which_model == 1:
                return 6
        if which_backward == 1:
            if which_model == 2:
                return 7
            if which_model == 1:
                return 8
    return 0

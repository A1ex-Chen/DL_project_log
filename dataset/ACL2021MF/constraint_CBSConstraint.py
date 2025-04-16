def CBSConstraint(CBS_type, max_constrain_num):
    if CBS_type == 'Two':
        assert max_constrain_num <= 2
        return TwoConstraint()
    elif CBS_type == 'GBS':
        return GBSConstraint(max_constrain_num)
    else:
        raise NotImplementedError

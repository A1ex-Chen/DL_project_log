def _group_str(names: List[str]) ->str:
    """
    Turn "common1", "common2", "common3" into "common{1,2,3}"
    """
    lcp = _longest_common_prefix_str(names)
    rest = [x[len(lcp):] for x in names]
    rest = '{' + ','.join(rest) + '}'
    ret = lcp + rest
    ret = ret.replace('bn_{beta,running_mean,running_var,gamma}', 'bn_*')
    ret = ret.replace('bn_beta,bn_running_mean,bn_running_var,bn_gamma', 'bn_*'
        )
    return ret

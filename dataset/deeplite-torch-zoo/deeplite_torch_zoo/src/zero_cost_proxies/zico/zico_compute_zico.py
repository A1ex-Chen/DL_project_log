def compute_zico(grad_dict, mode='sum'):
    for modname in grad_dict:
        grad_dict[modname] = np.array(grad_dict[modname])
    if mode not in ('sum', 'mean'):
        raise ValueError(
            f'`mode` argument for the ZiCo metric should be one of (`sum`, `mean`), but got {mode}'
            )
    nsr_mean_abs_agg = []
    for modname in grad_dict:
        nsr_std = np.std(grad_dict[modname], axis=0)
        nonzero_idx = np.nonzero(nsr_std)[0]
        nsr_mean_abs = np.mean(np.abs(grad_dict[modname]), axis=0)
        tmpsum = np.sum(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx])
        if tmpsum != 0:
            if mode == 'sum':
                nsr_mean_abs_agg.append(np.log(tmpsum))
            elif mode == 'mean':
                nsr_mean_abs_agg.append(np.log(np.mean(nsr_mean_abs[
                    nonzero_idx] / nsr_std[nonzero_idx])))
    return nsr_mean_abs_agg

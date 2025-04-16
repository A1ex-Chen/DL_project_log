def interp_t(trans, input_times, target_times):
    target_trans = []
    for target_t in target_times:
        diff = target_t - input_times
        array1 = diff.copy()
        array1[diff < 0] = 1000
        array2 = diff.copy()
        array2[diff > 0] = -1000
        t1_idx = np.argmin(array1)
        t2_idx = np.argmin(-array2)
        target_tran = (target_t - input_times[t1_idx]) / (input_times[
            t2_idx] - input_times[t1_idx]) * trans[t1_idx] + (input_times[
            t2_idx] - target_t) / (input_times[t2_idx] - input_times[t1_idx]
            ) * trans[t2_idx]
        target_trans.append(target_tran)
    target_trans = torch.stack(target_trans, axis=0)
    return target_trans

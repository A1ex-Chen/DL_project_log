def print_abs_stats(task_name, alpha, num_true, num_false, num_abstain, max_abs
    ):
    total = num_true + num_false
    tot_pred = total - num_abstain
    abs_frac = num_abstain / total
    abs_acc = 1.0
    if tot_pred > 0:
        abs_acc = num_true / tot_pred
    print(
        '        task,       alpha,     true,    false,  abstain,    total, tot_pred,   abs_frac,    max_abs,    abs_acc'
        )
    print(
        '{:>12s}, {:10.5e}, {:8d}, {:8d}, {:8d}, {:8d}, {:8d}, {:10.5f}, {:10.5f}, {:10.5f}'
        .format(task_name, alpha, num_true, num_false - num_abstain,
        num_abstain, total, tot_pred, abs_frac, max_abs, abs_acc))

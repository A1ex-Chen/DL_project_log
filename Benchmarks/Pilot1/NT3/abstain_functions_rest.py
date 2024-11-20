from tensorflow.keras import backend as K

abs_definitions = [
    {
        "name": "add_class",
        "nargs": "+",
        "type": int,
        "help": "flag to add abstention (per task)",
    },
    {
        "name": "alpha",
        "nargs": "+",
        "type": float,
        "help": "abstention penalty coefficient (per task)",
    },
    {
        "name": "min_acc",
        "nargs": "+",
        "type": float,
        "help": "minimum accuracy required (per task)",
    },
    {
        "name": "max_abs",
        "nargs": "+",
        "type": float,
        "help": "maximum abstention fraction allowed (per task)",
    },
    {
        "name": "alpha_scale_factor",
        "nargs": "+",
        "type": float,
        "help": "scaling factor for modifying alpha (per task)",
    },
    {
        "name": "init_abs_epoch",
        "action": "store",
        "type": int,
        "help": "number of epochs to skip before modifying alpha",
    },
    {
        "name": "n_iters",
        "action": "store",
        "type": int,
        "help": "number of iterations to iterate alpha",
    },
    {
        "name": "acc_gain",
        "type": float,
        "default": 5.0,
        "help": "factor to weight accuracy when determining new alpha scale",
    },
    {
        "name": "abs_gain",
        "type": float,
        "default": 1.0,
        "help": "factor to weight abstention fraction when determining new alpha scale",
    },
    {
        "name": "task_list",
        "nargs": "+",
        "type": int,
        "help": "list of task indices to use",
    },
    {
        "name": "task_names",
        "nargs": "+",
        "type": int,
        "help": "list of names corresponding to each task to use",
    },
    {"name": "cf_noise", "type": str, "help": "input file with cf noise"},
]









    return loss


def print_abs_stats(task_name, alpha, num_true, num_false, num_abstain, max_abs):

    # Compute interesting values
    total = num_true + num_false
    tot_pred = total - num_abstain
    abs_frac = num_abstain / total
    abs_acc = 1.0
    if tot_pred > 0:
        abs_acc = num_true / tot_pred

    print(
        "        task,       alpha,     true,    false,  abstain,    total, tot_pred,   abs_frac,    max_abs,    abs_acc"
    )
    print(
        "{:>12s}, {:10.5e}, {:8d}, {:8d}, {:8d}, {:8d}, {:8d}, {:10.5f}, {:10.5f}, {:10.5f}".format(
            task_name,
            alpha,
            num_true,
            num_false - num_abstain,
            num_abstain,
            total,
            tot_pred,
            abs_frac,
            max_abs,
            abs_acc,
        )
    )


def write_abs_stats(stats_file, alphas, accs, abst):

    # Open file for appending
    abs_file = open(stats_file, "a")

    # we write all the results
    for k in range((alphas.shape[0])):
        abs_file.write("%10.5e," % K.get_value(alphas[k]))
    for k in range((alphas.shape[0])):
        abs_file.write("%10.5e," % accs[k])
    for k in range((alphas.shape[0])):
        abs_file.write("%10.5e," % abst[k])
    abs_file.write("\n")
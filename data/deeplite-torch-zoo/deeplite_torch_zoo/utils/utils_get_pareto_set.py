def get_pareto_set(variable1, variable2, ignore_indices=None):
    array = np.array([variable1, variable2]).T
    sorting_indices = array[:, 0].argsort()
    if ignore_indices is not None:
        sorting_indices = [idx for idx in sorting_indices if idx not in
            ignore_indices]
    array = array[sorting_indices]
    ind_list = []
    pareto_frontier = array[0:1, :]
    ind_list.append(sorting_indices[0])
    for i, row in enumerate(array[1:, :]):
        if sum(row[x] >= pareto_frontier[-1][x] for x in range(len(row))
            ) == len(row):
            pareto_frontier = np.concatenate((pareto_frontier, [row]))
            ind_list.append(sorting_indices[i + 1])
    return ind_list

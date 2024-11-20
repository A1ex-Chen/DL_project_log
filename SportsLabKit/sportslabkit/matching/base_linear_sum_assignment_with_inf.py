def linear_sum_assignment_with_inf(cost_matrix: np.ndarray) ->tuple[np.
    ndarray, np.ndarray]:
    """Solve the linear sum assignment problem with inf values.

    Args:
        cost_matrix (np.ndarray): The cost matrix to solve.

    Raises:
        ValueError: Raises an error if the cost matrix contains both inf and -inf.
        ValueError: Raises an error if the cost matrix contains only inf or -inf.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The row and column indices of the assignment.
    """
    cost_matrix = np.asarray(cost_matrix)
    if cost_matrix.size == 0:
        return np.empty((0,), dtype=int), np.empty((0,), dtype=int)
    min_inf = np.isneginf(cost_matrix).any()
    max_inf = np.isposinf(cost_matrix).any()
    if min_inf and max_inf:
        raise ValueError('matrix contains both inf and -inf')
    if min_inf or max_inf:
        values = cost_matrix[~np.isinf(cost_matrix)]
        if values.size == 0:
            return np.empty((0,), dtype=int), np.empty((0,), dtype=int)
        if max_inf:
            place_holder = np.finfo(cost_matrix.dtype).max
        if min_inf:
            place_holder = np.finfo(cost_matrix.dtype).min
        cost_matrix[np.isinf(cost_matrix)] = place_holder
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    if min_inf or max_inf:
        valid_indices = cost_matrix[row_ind, col_ind] != place_holder
        return row_ind[valid_indices], col_ind[valid_indices]
    return row_ind, col_ind

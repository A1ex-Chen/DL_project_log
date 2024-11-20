def witness(X, gamma=1.0 / 128, L_0=64):
    """
      This function computes the persistence intervals for the dataset
      X using the witness complex.

    Args:
      X: 2d array representing the dataset.
      gamma: parameter determining the maximal persistence value.
      L_0: int, number of landmarks to use.

    Returns
      A list of persistence intervals and the maximal persistence value.
    """
    L = random_landmarks(X, L_0)
    W = X
    lmrk_tab, max_dist = lmrk_table(W, L)
    wc = gudhi.WitnessComplex(lmrk_tab)
    alpha_max = max_dist * gamma
    st = wc.create_simplex_tree(max_alpha_square=alpha_max, limit_dimension=2)
    st.persistence(homology_coeff_field=2)
    diag = st.persistence_intervals_in_dimension(1)
    return diag, alpha_max

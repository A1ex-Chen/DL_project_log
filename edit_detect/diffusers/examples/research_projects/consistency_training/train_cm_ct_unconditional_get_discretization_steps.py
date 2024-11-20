def get_discretization_steps(global_step: int, max_train_steps: int, s_0:
    int=10, s_1: int=1280, constant=False):
    """
    Calculates the current discretization steps at global step k using the discretization curriculum N(k).
    """
    if constant:
        return s_0 + 1
    k_prime = math.floor(max_train_steps / (math.log2(math.floor(s_1 / s_0)
        ) + 1))
    num_discretization_steps = min(s_0 * 2 ** math.floor(global_step /
        k_prime), s_1) + 1
    return num_discretization_steps

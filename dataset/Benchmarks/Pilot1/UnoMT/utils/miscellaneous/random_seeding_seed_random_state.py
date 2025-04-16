def seed_random_state(rand_state: int=0):
    """seed_random_state(0)

    This function sets up with random seed in multiple libraries possibly used
    during PyTorch training and validation.

    Args:
        rand_state (int): random seed

    Returns:
        None
    """
    random.seed(rand_state)
    np.random.seed(rand_state)
    torch.manual_seed(rand_state)
    torch.cuda.manual_seed_all(rand_state)
    torch.backends.cudnn.deterministic = True

def setup_seeds(master_seed, epochs, device):
    """
    Generates seeds from one master_seed.
    Function returns (worker_seeds, shuffling_seeds), worker_seeds are later
    used to initialize per-worker random number generators (mostly for
    dropouts), shuffling_seeds are for RNGs resposible for reshuffling the
    dataset before each epoch.
    Seeds are generated on worker with rank 0 and broadcasted to all other
    workers.

    :param master_seed: master RNG seed used to initialize other generators
    :param epochs: number of epochs
    :param device: torch.device (used for distributed.broadcast)
    """
    if master_seed is None:
        master_seed = random.SystemRandom().randint(0, 2 ** 32 - 1)
        if get_rank() == 0:
            logging.info(f'Using random master seed: {master_seed}')
    else:
        logging.info(f'Using master seed from command line: {master_seed}')
    seeding_rng = random.Random(master_seed)
    worker_seeds = generate_seeds(seeding_rng, get_world_size())
    shuffling_seeds = generate_seeds(seeding_rng, epochs)
    worker_seeds = broadcast_seeds(worker_seeds, device)
    shuffling_seeds = broadcast_seeds(shuffling_seeds, device)
    return worker_seeds, shuffling_seeds

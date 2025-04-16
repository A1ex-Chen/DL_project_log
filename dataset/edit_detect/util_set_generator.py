def set_generator(generator: Union[int, torch.Generator]) ->torch.Generator:
    if isinstance(generator, int):
        rng: torch.Generator = torch.Generator()
        rng.manual_seed(generator)
    elif isinstance(generator, torch.Generator):
        rng = generator
    return rng

def first_n(n, generator):
    for i, d in zip(range(n), generator):
        yield d

def map(x):
    if x == -1:
        return -1
    else:
        rnd = random.uniform(0, 1)
        if rnd < 0.05:
            return dd[random.randint(1, ent_size)]
        elif rnd < 0.2:
            return -1
        else:
            return x

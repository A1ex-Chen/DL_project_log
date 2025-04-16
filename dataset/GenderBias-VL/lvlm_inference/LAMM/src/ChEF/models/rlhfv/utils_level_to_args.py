def level_to_args(level):
    level = level / MAX_LEVEL * 30
    if np.random.random() < 0.5:
        level = -level
    return level, replace_value

def shear_level_to_args(MAX_LEVEL, replace_value):

    def level_to_args(level):
        level = level / MAX_LEVEL * 0.3
        if np.random.random() > 0.5:
            level = -level
        return level, replace_value
    return level_to_args

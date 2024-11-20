def translate_level_to_args(translate_const, MAX_LEVEL, replace_value):

    def level_to_args(level):
        level = level / MAX_LEVEL * float(translate_const)
        if np.random.random() > 0.5:
            level = -level
        return level, replace_value
    return level_to_args

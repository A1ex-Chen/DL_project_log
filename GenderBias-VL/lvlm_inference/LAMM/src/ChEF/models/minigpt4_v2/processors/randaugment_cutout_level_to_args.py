def cutout_level_to_args(cutout_const, MAX_LEVEL, replace_value):

    def level_to_args(level):
        level = int(level / MAX_LEVEL * cutout_const)
        return level, replace_value
    return level_to_args

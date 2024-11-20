def solarize_level_to_args(MAX_LEVEL):

    def level_to_args(level):
        level = int(level / MAX_LEVEL * 256)
        return level,
    return level_to_args

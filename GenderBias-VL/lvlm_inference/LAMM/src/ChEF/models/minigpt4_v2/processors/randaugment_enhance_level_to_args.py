def enhance_level_to_args(MAX_LEVEL):

    def level_to_args(level):
        return level / MAX_LEVEL * 1.8 + 0.1,
    return level_to_args

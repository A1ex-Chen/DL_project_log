def build_inferencer(inferencer_type, **kwargs):
    return inferencer_dict[inferencer_type](**kwargs)

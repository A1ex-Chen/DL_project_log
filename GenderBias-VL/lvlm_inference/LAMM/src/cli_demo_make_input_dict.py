def make_input_dict(args, vision_path):
    input_dict = dict()
    for key in INPUT_KEYS:
        if key.split('_')[0] == args.vision_type:
            input_dict[key] = vision_path
        else:
            input_dict[key] = None
    return input_dict

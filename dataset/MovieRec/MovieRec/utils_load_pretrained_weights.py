def load_pretrained_weights(model, path):
    chk_dict = torch.load(os.path.abspath(path))
    model_state_dict = chk_dict[STATE_DICT_KEY
        ] if STATE_DICT_KEY in chk_dict else chk_dict['state_dict']
    model.load_state_dict(model_state_dict)

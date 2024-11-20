def load_states_from_checkpoint(model_file: str) ->CheckpointState:
    print('Reading saved model from %s', model_file)
    state_dict = torch.load(model_file, map_location=lambda s, l:
        default_restore_location(s, 'cpu'))
    return CheckpointState(**state_dict)

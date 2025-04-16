def get_model_state_files(checkpoint_dir):
    return get_checkpoint_files(checkpoint_dir, '*_model_states.pt')

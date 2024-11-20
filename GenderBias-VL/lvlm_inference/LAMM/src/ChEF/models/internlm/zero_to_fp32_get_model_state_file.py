def get_model_state_file(checkpoint_dir, zero_stage):
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Directory '{checkpoint_dir}' doesn't exist")
    if zero_stage <= 2:
        file = os.path.join(checkpoint_dir, 'mp_rank_00_model_states.pt')
    elif zero_stage == 3:
        file = os.path.join(checkpoint_dir,
            'zero_pp_rank_0_mp_rank_00_model_states.pt')
    if not os.path.exists(file):
        raise FileNotFoundError(f"can't find model states file at '{file}'")
    return file

def convert_projection_checkpoint(checkpoint):
    projection_state_dict = {}
    conditioner_state_dict = extract_sub_model(checkpoint, key_prefix=
        'cond_stage_models.0.')
    projection_state_dict['sos_embed'] = conditioner_state_dict[
        'start_of_sequence_tokens.weight'][0]
    projection_state_dict['sos_embed_1'] = conditioner_state_dict[
        'start_of_sequence_tokens.weight'][1]
    projection_state_dict['eos_embed'] = conditioner_state_dict[
        'end_of_sequence_tokens.weight'][0]
    projection_state_dict['eos_embed_1'] = conditioner_state_dict[
        'end_of_sequence_tokens.weight'][1]
    projection_state_dict['projection.weight'] = conditioner_state_dict[
        'input_sequence_embed_linear.0.weight']
    projection_state_dict['projection.bias'] = conditioner_state_dict[
        'input_sequence_embed_linear.0.bias']
    projection_state_dict['projection_1.weight'] = conditioner_state_dict[
        'input_sequence_embed_linear.1.weight']
    projection_state_dict['projection_1.bias'] = conditioner_state_dict[
        'input_sequence_embed_linear.1.bias']
    return projection_state_dict

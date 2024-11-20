def encode_prompt(text_encoder, input_ids):
    outputs = text_encoder(input_ids, return_dict=True,
        output_hidden_states=True)
    encoder_hidden_states = outputs.hidden_states[-2]
    cond_embeds = outputs[0]
    return encoder_hidden_states, cond_embeds

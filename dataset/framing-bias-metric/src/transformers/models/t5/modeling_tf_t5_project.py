def project(hidden_states, proj_layer, key_value_states, past_key_value):
    """ projects hidden states correctly to key/query states """
    if key_value_states is None:
        hidden_states = shape(proj_layer(hidden_states))
    elif past_key_value is None:
        hidden_states = shape(proj_layer(key_value_states))
    if past_key_value is not None:
        if key_value_states is None:
            hidden_states = tf.concat([past_key_value, hidden_states], axis=2)
        else:
            hidden_states = past_key_value
    return hidden_states

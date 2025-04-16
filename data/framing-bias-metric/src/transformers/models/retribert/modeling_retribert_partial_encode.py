def partial_encode(*inputs):
    encoder_outputs = sent_encoder.encoder(inputs[0], attention_mask=inputs
        [1], head_mask=head_mask)
    sequence_output = encoder_outputs[0]
    pooled_output = sent_encoder.pooler(sequence_output)
    return pooled_output

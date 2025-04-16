def ngram_attention_bias(sequence_length, ngram, device, dtype):
    """
    This function computes the bias for the predict stream
    """
    bias = torch.ones((ngram, sequence_length, 2 * sequence_length), device
        =device, dtype=dtype) * float('-inf')
    for stream_idx in range(ngram):
        for i in range(sequence_length):
            bias[stream_idx, i, sequence_length + i] = 0
            bias[stream_idx, i, :max(i - stream_idx, 0) + 1] = 0
    return bias

def new_synthesize_flattened_rnn_weights(fp32_weights, fp16_flat_tensor,
    rnn_fn='', verbose=False):
    fp16_weights = []
    fp32_base_ptr = fp32_weights[0].data_ptr()
    for w_fp32 in fp32_weights:
        w_fp16 = w_fp32.new().half()
        offset = (w_fp32.data_ptr() - fp32_base_ptr) // w_fp32.element_size()
        w_fp16.set_(fp16_flat_tensor.storage(), offset, w_fp32.shape)
        w_fp16.copy_(w_fp32)
        if verbose:
            print('Float->Half ({})'.format(rnn_fn))
        fp16_weights.append(w_fp16)
    return fp16_weights

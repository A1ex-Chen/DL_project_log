def extend_enc_output(tensor, num_beams=None):
    tensor = tensor[None, None, :].reshape((batch_size, 1, n_docs) + tensor
        .shape[1:])
    tensor = tensor.expand((batch_size, num_beams, n_docs) + tensor.shape[3:])
    return tensor.reshape((batch_size * num_beams * n_docs,) + tensor.shape[3:]
        )

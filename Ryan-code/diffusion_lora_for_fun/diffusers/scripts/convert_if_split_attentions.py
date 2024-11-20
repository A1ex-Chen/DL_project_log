def split_attentions(*, weight, bias, split, chunk_size):
    weights = [None] * split
    biases = [None] * split
    weights_biases_idx = 0
    for starting_row_index in range(0, weight.shape[0], chunk_size):
        row_indices = torch.arange(starting_row_index, starting_row_index +
            chunk_size)
        weight_rows = weight[row_indices, :]
        bias_rows = bias[row_indices]
        if weights[weights_biases_idx] is None:
            weights[weights_biases_idx] = weight_rows
            biases[weights_biases_idx] = bias_rows
        else:
            assert weights[weights_biases_idx] is not None
            weights[weights_biases_idx] = torch.concat([weights[
                weights_biases_idx], weight_rows])
            biases[weights_biases_idx] = torch.concat([biases[
                weights_biases_idx], bias_rows])
        weights_biases_idx = (weights_biases_idx + 1) % split
    return weights, biases

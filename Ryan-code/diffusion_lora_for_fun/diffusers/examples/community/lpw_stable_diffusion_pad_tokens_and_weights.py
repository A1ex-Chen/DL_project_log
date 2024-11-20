def pad_tokens_and_weights(tokens, weights, max_length, bos, eos, pad,
    no_boseos_middle=True, chunk_length=77):
    """
    Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
    """
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    weights_length = (max_length if no_boseos_middle else 
        max_embeddings_multiples * chunk_length)
    for i in range(len(tokens)):
        tokens[i] = [bos] + tokens[i] + [pad] * (max_length - 1 - len(
            tokens[i]) - 1) + [eos]
        if no_boseos_middle:
            weights[i] = [1.0] + weights[i] + [1.0] * (max_length - 1 - len
                (weights[i]))
        else:
            w = []
            if len(weights[i]) == 0:
                w = [1.0] * weights_length
            else:
                for j in range(max_embeddings_multiples):
                    w.append(1.0)
                    w += weights[i][j * (chunk_length - 2):min(len(weights[
                        i]), (j + 1) * (chunk_length - 2))]
                    w.append(1.0)
                w += [1.0] * (weights_length - len(w))
            weights[i] = w[:]
    return tokens, weights

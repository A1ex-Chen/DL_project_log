def encode_position(input, levels, inc_input):
    """
    For each scalar, we encode it using a series of sin() and cos() functions with different frequency.
        - With L pairs of sin/cos function, each scalar is encoded to a vector that has 2L elements. Concatenating with
          itself results in 2L+1 elements.
        - With C channels, we get C(2L+1) channels output.

    :param input:   (..., C)            torch.float32
    :param levels:  scalar L            int
    :return:        (..., C*(2L+1))     torch.float32
    """
    result_list = [input] if inc_input else []
    for i in range(levels):
        temp = 2.0 ** i * input
        result_list.append(torch.sin(temp))
        result_list.append(torch.cos(temp))
    result_list = torch.cat(result_list, dim=-1)
    return result_list

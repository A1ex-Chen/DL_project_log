def run_conv2d(x, weights, bias):
    n_conv = len(weights)
    for i in range(n_conv):
        x = F.conv2d(x, weights[i], bias[i])
        if i != n_conv - 1:
            x = F.relu(x)
    return x

def jit_relu_dropout(x, prob):
    x = torch.nn.functional.relu(x)
    x = torch.nn.functional.dropout(x, p=prob, training=True)
    return x

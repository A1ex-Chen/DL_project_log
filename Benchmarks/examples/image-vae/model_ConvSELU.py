def ConvSELU(i, o, kernel_size=3, padding=0, p=0.0):
    model = [nn.Conv1d(i, o, kernel_size=kernel_size, padding=padding),
        SELU(inplace=True)]
    if p > 0.0:
        model += [nn.Dropout(p)]
    return nn.Sequential(*model)

def get_activation(name):

    def hook(model, input, output):
        activations[name] = output
    return hook

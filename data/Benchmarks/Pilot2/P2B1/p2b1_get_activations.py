def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()
        ], [model.layers[layer].output])
    activations = get_activations([X_batch, 0])
    return activations

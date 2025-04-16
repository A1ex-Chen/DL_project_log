def get_output_embeddings(self):
    """Double-check if you are using adaptive softmax."""
    if self.sample_softmax > 0:
        return self.out_layer
    else:
        return self.crit.out_layers[-1]

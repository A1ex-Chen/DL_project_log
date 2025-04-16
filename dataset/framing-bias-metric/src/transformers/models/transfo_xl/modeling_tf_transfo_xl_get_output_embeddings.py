def get_output_embeddings(self):
    """Double-check if you are using adaptive softmax."""
    if len(self.crit.out_layers) > 0:
        return self.crit.out_layers[-1]
    return None

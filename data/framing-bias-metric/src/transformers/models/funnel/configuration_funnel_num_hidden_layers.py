@property
def num_hidden_layers(self):
    return sum(self.block_sizes)

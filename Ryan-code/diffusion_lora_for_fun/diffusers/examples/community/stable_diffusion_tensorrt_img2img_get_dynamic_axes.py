def get_dynamic_axes(self):
    return {'images': {(0): 'B', (2): '8H', (3): '8W'}, 'latent': {(0): 'B',
        (2): 'H', (3): 'W'}}

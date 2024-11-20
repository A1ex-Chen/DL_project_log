@property
def dummy_renderer(self):
    torch.manual_seed(0)
    model_kwargs = {'param_shapes': ((self.renderer_dim, 93), (self.
        renderer_dim, 8), (self.renderer_dim, 8), (self.renderer_dim, 8)),
        'd_latent': self.time_input_dim, 'd_hidden': self.renderer_dim,
        'n_output': 12, 'background': (0.1, 0.1, 0.1)}
    model = ShapERenderer(**model_kwargs)
    return model

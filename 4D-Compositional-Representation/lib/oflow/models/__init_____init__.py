def __init__(self, decoder, encoder=None, encoder_latent=None,
    encoder_latent_temporal=None, encoder_temporal=None, vector_field=None,
    ode_step_size=None, use_adjoint=False, rtol=0.001, atol=1e-05,
    ode_solver='dopri5', p0_z=None, device=None, input_type=None, **kwargs):
    super().__init__()
    if p0_z is None:
        p0_z = dist.Normal(torch.tensor([]), torch.tensor([]))
    self.device = device
    self.input_type = input_type
    self.decoder = decoder
    self.encoder_latent = encoder_latent
    self.encoder_latent_temporal = encoder_latent_temporal
    self.encoder = encoder
    self.vector_field = vector_field
    self.encoder_temporal = encoder_temporal
    self.p0_z = p0_z
    self.rtol = rtol
    self.atol = atol
    self.ode_solver = ode_solver
    if use_adjoint:
        self.odeint = odeint_adjoint
    else:
        self.odeint = odeint
    self.ode_options = {}
    if ode_step_size:
        self.ode_options['step_size'] = ode_step_size

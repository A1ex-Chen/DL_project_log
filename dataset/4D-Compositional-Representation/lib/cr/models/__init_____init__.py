def __init__(self, decoder, encoder_latent=None, vector_field=None, encoder
    =None, encoder_motion=None, encoder_identity=None, ode_step_size=None,
    use_adjoint=False, rtol=0.001, atol=1e-05, ode_solver='dopri5', device=
    None, input_type=None, **kwargs):
    super().__init__()
    self.device = device
    self.input_type = input_type
    self.decoder = decoder
    self.encoder_latent = encoder_latent
    self.encoder = encoder
    self.encoder_motion = encoder_motion
    self.encoder_identity = encoder_identity
    self.vector_field = vector_field
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

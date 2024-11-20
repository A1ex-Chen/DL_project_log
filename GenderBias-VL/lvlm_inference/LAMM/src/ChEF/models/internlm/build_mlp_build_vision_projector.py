def build_vision_projector():
    projector_type = 'mlp2x_gelu'
    mm_hidden_size = 1024
    hidden_size = 4096
    mlp_gelu_match = re.match('^mlp(\\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(mm_hidden_size, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        return nn.Sequential(*modules)
    if projector_type == 'identity':
        return IdentityMap()
    raise ValueError(f'Unknown projector type: {projector_type}')

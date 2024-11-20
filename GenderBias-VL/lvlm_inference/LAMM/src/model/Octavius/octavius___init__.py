def __init__(self, **args):
    args['vision_type'] = args['octavius_modality']
    assert len(args['vision_type']) > 0
    super().__init__(**args)
    print(f'Octavius Modalities: {self.vision_type}')
    if 'image' in self.vision_type:
        self.llama_proj = nn.Linear(self.vision_hidden_size, self.
            llama_model.config.hidden_size)
        print('Octavius 2D projection layer initialized.')
    if 'pcl' in self.vision_type:
        self.resampler_3d = Resampler3D(args['num_query_rsp_3d'], args[
            'num_heads_rsp_3d'], args['num_layers_rsp_3d'], args[
            'hidden_size_rsp_3d'])
        self.pos_3d_proj = nn.Sequential(nn.Linear(6, 256), nn.ReLU(), nn.
            Linear(256, args['hidden_size_rsp_3d']))
        self.llama_proj_3d = nn.Sequential(nn.Linear(args[
            'hidden_size_rsp_3d'], 5120), nn.LayerNorm(5120))
        print('Octavius 3D Encoder & projection layer initialized.')
    self.gate_mode = self.args['moe_gate_mode']
    assert self.gate_mode in ['top2_gate']
    self.num_experts = self.args['moe_lora_num_experts']
    self.gating_network = Top2Gating(self.llama_model.config.hidden_size,
        self.num_experts)
    self.device = torch.cuda.current_device()

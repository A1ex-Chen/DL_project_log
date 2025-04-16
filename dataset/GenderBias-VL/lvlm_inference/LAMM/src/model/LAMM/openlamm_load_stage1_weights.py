def load_stage1_weights(self, ckpt_path):
    original_state_dict = torch.load(ckpt_path)
    lm_head_weights = {}
    llama_proj_weights = {}
    for key, value in original_state_dict.items():
        if key.startswith('llama_model.lm_head'):
            lm_head_weights[key.split('.')[-1]] = value
        elif key.startswith('llama_proj'):
            llama_proj_weights[key.split('.')[-1]] = value
    self.llama_proj.load_state_dict(llama_proj_weights)
    self.llama_model.lm_head.load_state_dict(lm_head_weights)

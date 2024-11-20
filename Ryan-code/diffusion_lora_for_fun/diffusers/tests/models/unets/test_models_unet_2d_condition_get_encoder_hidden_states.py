def get_encoder_hidden_states(self, seed=0, shape=(4, 77, 768), fp16=False):
    dtype = torch.float16 if fp16 else torch.float32
    hidden_states = torch.from_numpy(load_hf_numpy(self.get_file_format(
        seed, shape))).to(torch_device).to(dtype)
    return hidden_states

def simulate_backprop(self, unet):
    updated_state_dict = {}
    for k, param in unet.state_dict().items():
        updated_param = torch.randn_like(param) + param * torch.randn_like(
            param)
        updated_state_dict.update({k: updated_param})
    unet.load_state_dict(updated_state_dict)
    return unet

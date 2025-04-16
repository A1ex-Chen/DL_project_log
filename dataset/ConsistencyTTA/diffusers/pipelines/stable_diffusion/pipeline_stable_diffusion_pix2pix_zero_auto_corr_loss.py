def auto_corr_loss(self, hidden_states, generator=None):
    batch_size, channel, height, width = hidden_states.shape
    if batch_size > 1:
        raise ValueError('Only batch_size 1 is supported for now')
    hidden_states = hidden_states.squeeze(0)
    reg_loss = 0.0
    for i in range(hidden_states.shape[0]):
        noise = hidden_states[i][None, None, :, :]
        while True:
            roll_amount = torch.randint(noise.shape[2] // 2, (1,),
                generator=generator).item()
            reg_loss += (noise * torch.roll(noise, shifts=roll_amount, dims=2)
                ).mean() ** 2
            reg_loss += (noise * torch.roll(noise, shifts=roll_amount, dims=3)
                ).mean() ** 2
            if noise.shape[2] <= 8:
                break
            noise = F.avg_pool2d(noise, kernel_size=2)
    return reg_loss

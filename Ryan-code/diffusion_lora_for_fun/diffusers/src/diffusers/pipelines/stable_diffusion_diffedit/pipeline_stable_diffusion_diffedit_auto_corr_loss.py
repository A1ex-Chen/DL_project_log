def auto_corr_loss(hidden_states, generator=None):
    reg_loss = 0.0
    for i in range(hidden_states.shape[0]):
        for j in range(hidden_states.shape[1]):
            noise = hidden_states[i:i + 1, j:j + 1, :, :]
            while True:
                roll_amount = torch.randint(noise.shape[2] // 2, (1,),
                    generator=generator).item()
                reg_loss += (noise * torch.roll(noise, shifts=roll_amount,
                    dims=2)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=roll_amount,
                    dims=3)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = torch.nn.functional.avg_pool2d(noise, kernel_size=2)
    return reg_loss

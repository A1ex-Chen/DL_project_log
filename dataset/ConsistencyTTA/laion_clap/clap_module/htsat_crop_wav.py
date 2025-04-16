def crop_wav(self, x, crop_size, spe_pos=None):
    time_steps = x.shape[2]
    tx = torch.zeros(x.shape[0], x.shape[1], crop_size, x.shape[3]).to(x.device
        )
    for i in range(len(x)):
        if spe_pos is None:
            crop_pos = random.randint(0, time_steps - crop_size - 1)
        else:
            crop_pos = spe_pos
        tx[i][0] = x[i, 0, crop_pos:crop_pos + crop_size, :]
    return tx

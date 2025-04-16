def inverse(self, magnitude, phase):
    device = self.forward_basis.device
    magnitude, phase = magnitude.to(device), phase.to(device)
    recombine_magnitude_phase = torch.cat([magnitude * torch.cos(phase), 
        magnitude * torch.sin(phase)], dim=1)
    inverse_transform = F.conv_transpose1d(recombine_magnitude_phase, torch
        .autograd.Variable(self.inverse_basis, requires_grad=False), stride
        =self.hop_length, padding=0)
    if self.window is not None:
        window_sum = window_sumsquare(self.window, magnitude.size(-1),
            hop_length=self.hop_length, win_length=self.win_length, n_fft=
            self.filter_length, dtype=np.float32)
        approx_nonzero_indices = torch.from_numpy(np.where(window_sum >
            tiny(window_sum))[0])
        window_sum = torch.autograd.Variable(torch.from_numpy(window_sum),
            requires_grad=False)
        window_sum = window_sum
        inverse_transform[:, :, approx_nonzero_indices] /= window_sum[
            approx_nonzero_indices]
        inverse_transform *= float(self.filter_length) / self.hop_length
    inverse_transform = inverse_transform[:, :, int(self.filter_length / 2):]
    inverse_transform = inverse_transform[:, :, :-int(self.filter_length / 2)]
    return inverse_transform

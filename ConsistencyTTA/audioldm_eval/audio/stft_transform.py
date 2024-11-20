def transform(self, input_data):
    num_batches = input_data.size(0)
    num_samples = input_data.size(1)
    self.num_samples = num_samples
    input_data = input_data.view(num_batches, 1, num_samples)
    input_data_len = input_data.unsqueeze(1).shape[3]
    assert input_data_len > 10000, f'Length is {input_data_len}.'
    input_data = F.pad(input_data.unsqueeze(1), (int(self.filter_length / 2
        ), int(self.filter_length / 2), 0, 0), mode='reflect')
    input_data = input_data.squeeze(1)
    forward_transform = F.conv1d(input_data, torch.autograd.Variable(self.
        forward_basis, requires_grad=False), stride=self.hop_length, padding=0
        ).cpu()
    cutoff = int(self.filter_length / 2 + 1)
    real_part = forward_transform[:, :cutoff, :]
    imag_part = forward_transform[:, cutoff:, :]
    magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
    phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data)
        )
    return magnitude, phase

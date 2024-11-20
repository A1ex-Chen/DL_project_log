def convert(self, z):
    z = torch.cat(z, 1)
    box = z[:, :, :4]
    conf = z[:, :, 4:5]
    score = z[:, :, 5:]
    score *= conf
    convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 
        0.5, 0], [0, -0.5, 0, 0.5]], dtype=torch.float32, device=z.device)
    box @= convert_matrix
    return box, score

@parameterized.expand([[(1, 4, 73, 97), (1, 3, 584, 776)], [(1, 4, 97, 73),
    (1, 3, 776, 584)], [(1, 4, 49, 65), (1, 3, 392, 520)], [(1, 4, 65, 49),
    (1, 3, 520, 392)], [(1, 4, 49, 49), (1, 3, 392, 392)]])
def test_tae_tiling(self, in_shape, out_shape):
    model = self.get_sd_vae_model()
    model.enable_tiling()
    with torch.no_grad():
        zeros = torch.zeros(in_shape).to(torch_device)
        dec = model.decode(zeros).sample
        assert dec.shape == out_shape

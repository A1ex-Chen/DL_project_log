@parameterized.expand([[13, [-0.5861, 0.1283, -0.0931, 0.0882, 0.4476, 
    0.1329, -0.0498, 0.064]], [37, [-0.4913, 0.011, -0.0483, 0.0541, 0.4954,
    -0.017, 0.0354, 0.1651]]])
def test_kandinsky_prior(self, seed, expected_slice):
    model = PriorTransformer.from_pretrained(
        'kandinsky-community/kandinsky-2-1-prior', subfolder='prior')
    model.to(torch_device)
    input = self.get_dummy_seed_input(seed=seed)
    with torch.no_grad():
        sample = model(**input)[0]
    assert list(sample.shape) == [1, 768]
    output_slice = sample[0, :8].flatten().cpu()
    print(output_slice)
    expected_output_slice = torch.tensor(expected_slice)
    assert torch_all_close(output_slice, expected_output_slice, atol=0.001)

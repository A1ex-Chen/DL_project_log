@pytest.mark.slow
def test_inference_masked_lm(self):
    model = RobertaForMaskedLM.from_pretrained('roberta-base')
    input_ids = torch.tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 
        46078, 1588, 2]])
    output = model(input_ids)[0]
    expected_shape = torch.Size((1, 11, 50265))
    self.assertEqual(output.shape, expected_shape)
    expected_slice = torch.Tensor([[[33.8843, -4.3107, 22.7779], [4.6533, -
        2.8099, 13.6252], [1.8222, -3.6898, 8.86]]])
    self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=
        0.001))

@pytest.mark.slow
def test_inference_classification_head(self):
    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-large-mnli')
    input_ids = torch.tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 
        46078, 1588, 2]])
    output = model(input_ids)[0]
    expected_shape = torch.Size((1, 3))
    self.assertEqual(output.shape, expected_shape)
    expected_tensor = torch.Tensor([[-0.9469, 0.3913, 0.5118]])
    self.assertTrue(torch.allclose(output, expected_tensor, atol=0.001))

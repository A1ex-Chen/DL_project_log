@pytest.mark.slow
def test_inference_classification_head(self):
    model = TFRobertaForSequenceClassification.from_pretrained(
        'roberta-large-mnli')
    input_ids = tf.constant([[0, 31414, 232, 328, 740, 1140, 12695, 69, 
        46078, 1588, 2]])
    output = model(input_ids)[0]
    expected_shape = [1, 3]
    self.assertEqual(list(output.numpy().shape), expected_shape)
    expected_tensor = tf.constant([[-0.9469, 0.3913, 0.5118]])
    self.assertTrue(numpy.allclose(output.numpy(), expected_tensor.numpy(),
        atol=0.001))

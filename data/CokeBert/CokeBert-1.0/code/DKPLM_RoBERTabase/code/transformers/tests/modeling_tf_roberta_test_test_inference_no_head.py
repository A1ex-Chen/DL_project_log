@pytest.mark.slow
def test_inference_no_head(self):
    model = TFRobertaModel.from_pretrained('roberta-base')
    input_ids = tf.constant([[0, 31414, 232, 328, 740, 1140, 12695, 69, 
        46078, 1588, 2]])
    output = model(input_ids)[0]
    expected_slice = tf.constant([[[-0.0231, 0.0782, 0.0074], [-0.1854, 
        0.0539, -0.0174], [0.0548, 0.0799, 0.1687]]])
    self.assertTrue(numpy.allclose(output[:, :3, :3].numpy(),
        expected_slice.numpy(), atol=0.001))

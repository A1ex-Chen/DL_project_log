def test_output_pretrained(self):
    model = PriorTransformer.from_pretrained('hf-internal-testing/prior-dummy')
    model = model.to(torch_device)
    if hasattr(model, 'set_default_attn_processor'):
        model.set_default_attn_processor()
    input = self.get_dummy_seed_input()
    with torch.no_grad():
        output = model(**input)[0]
    output_slice = output[0, :5].flatten().cpu()
    print(output_slice)
    expected_output_slice = torch.tensor([-1.3436, -0.287, 0.7538, 0.4368, 
        -0.0239])
    self.assertTrue(torch_all_close(output_slice, expected_output_slice,
        rtol=0.01))

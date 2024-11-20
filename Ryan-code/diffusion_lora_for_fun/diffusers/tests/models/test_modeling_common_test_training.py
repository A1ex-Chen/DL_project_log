@require_torch_accelerator_with_training
def test_training(self):
    init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    model = self.model_class(**init_dict)
    model.to(torch_device)
    model.train()
    output = model(**inputs_dict)
    if isinstance(output, dict):
        output = output.to_tuple()[0]
    input_tensor = inputs_dict[self.main_input_name]
    noise = torch.randn((input_tensor.shape[0],) + self.output_shape).to(
        torch_device)
    loss = torch.nn.functional.mse_loss(output, noise)
    loss.backward()

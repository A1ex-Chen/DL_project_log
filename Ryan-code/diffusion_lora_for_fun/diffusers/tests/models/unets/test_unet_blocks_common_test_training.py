@require_torch_accelerator_with_training
def test_training(self):
    init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    model = self.block_class(**init_dict)
    model.to(torch_device)
    model.train()
    output = model(**inputs_dict)
    if isinstance(output, Tuple):
        output = output[0]
    device = torch.device(torch_device)
    noise = randn_tensor(output.shape, device=device)
    loss = torch.nn.functional.mse_loss(output, noise)
    loss.backward()

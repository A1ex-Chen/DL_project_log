@require_torch_accelerator_with_training
def test_gradient_checkpointing(self):
    init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    model = self.model_class(**init_dict)
    model.to(torch_device)
    assert not model.is_gradient_checkpointing and model.training
    out = model(**inputs_dict).sample
    model.zero_grad()
    labels = torch.randn_like(out)
    loss = (out - labels).mean()
    loss.backward()
    model_2 = self.model_class(**init_dict)
    model_2.load_state_dict(model.state_dict())
    model_2.to(torch_device)
    model_2.enable_gradient_checkpointing()
    assert model_2.is_gradient_checkpointing and model_2.training
    out_2 = model_2(**inputs_dict).sample
    model_2.zero_grad()
    loss_2 = (out_2 - labels).mean()
    loss_2.backward()
    self.assertTrue((loss - loss_2).abs() < 1e-05)
    named_params = dict(model.named_parameters())
    named_params_2 = dict(model_2.named_parameters())
    for name, param in named_params.items():
        self.assertTrue(torch_all_close(param.grad.data, named_params_2[
            name].grad.data, atol=5e-05))

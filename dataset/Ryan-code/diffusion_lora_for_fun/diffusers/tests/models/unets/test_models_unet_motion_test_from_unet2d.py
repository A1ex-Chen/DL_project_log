def test_from_unet2d(self):
    torch.manual_seed(0)
    unet2d = UNet2DConditionModel()
    torch.manual_seed(1)
    model = self.model_class.from_unet2d(unet2d)
    model_state_dict = model.state_dict()
    for param_name, param_value in unet2d.named_parameters():
        self.assertTrue(torch.equal(model_state_dict[param_name], param_value))

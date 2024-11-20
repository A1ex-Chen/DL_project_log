def test_loss_scale_decrease(self):
    num_losses = 3
    nb_decrease_loss_scales = [0, 1, 2]
    for opt_level in self.test_opt_levels:
        nb_decrease_loss_scales_tmp = list(nb_decrease_loss_scales)
        model = MyModel().to('cuda')
        optimizer = optim.SGD(model.parameters(), lr=self.initial_lr)
        model, optimizer = amp.initialize(model, optimizer, opt_level=
            opt_level, num_losses=num_losses, verbosity=0)
        if amp._amp_state.opt_properties.loss_scale != 'dynamic':
            continue
        initial_loss_scales = []
        for idx in range(num_losses):
            initial_loss_scales.append(amp._amp_state.loss_scalers[idx].
                loss_scale())
        for _ in range(len(nb_decrease_loss_scales)):
            x = torch.randn(16, 3, 24, 24, device='cuda')
            for idx in range(num_losses):
                while nb_decrease_loss_scales_tmp[idx] > 0:
                    optimizer.zero_grad()
                    output = model(x * 2 ** 17)
                    loss = output.mean()
                    with amp.scale_loss(loss, optimizer, loss_id=idx
                        ) as scaled_loss:
                        scaled_loss.backward(retain_graph=True)
                    optimizer.step()
                    nb_decrease_loss_scales_tmp[idx] -= 1
        updated_loss_scales = []
        for idx in range(num_losses):
            updated_loss_scales.append(amp._amp_state.loss_scalers[idx].
                loss_scale())
        for factor, update_ls, init_ls in zip(nb_decrease_loss_scales,
            updated_loss_scales, initial_loss_scales):
            self.assertEqual(update_ls, init_ls / 2 ** factor)
        amp_state_dict = amp.state_dict()
        for scaler_idx, factor, init_ls in zip(amp_state_dict,
            nb_decrease_loss_scales, initial_loss_scales):
            scaler = amp_state_dict[scaler_idx]
            self.assertEqual(scaler['loss_scale'], init_ls / 2 ** factor)
            unskipped_target = 0
            self.assertEqual(scaler['unskipped'], unskipped_target)

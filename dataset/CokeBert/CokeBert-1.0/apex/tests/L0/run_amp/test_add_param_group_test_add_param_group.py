def test_add_param_group(self):
    for opt_level in ('O0', 'O1', 'O2', 'O3'):
        for zero_before_add in (True, False):
            for try_accumulation in (True, False):
                model0 = MyModel(1)
                model1 = MyModel(2)
                optimizer = torch.optim.SGD([{'params': model0.parameters(),
                    'lr': 0.25}], momentum=0.125)
                optimizer.zero_grad()
                loss = model0(self.x)
                loss.backward()
                optimizer.step()
                if zero_before_add:
                    optimizer.zero_grad()
                optimizer.add_param_group({'params': model1.parameters(),
                    'lr': 0.5})
                if not zero_before_add:
                    optimizer.zero_grad()
                loss = model0(self.x) + model1(self.x)
                loss.backward(retain_graph=try_accumulation)
                if try_accumulation:
                    loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss = model0(self.x) + model1(self.x)
                loss.backward(retain_graph=try_accumulation)
                if try_accumulation:
                    loss.backward()
                optimizer.step()
                reference_params = [param.data.clone() for param in model0.
                    parameters()] + [param.data.clone() for param in model1
                    .parameters()]
                for how_to_zero in ('none', 'model', 'optimizer'):
                    model0 = MyModel(1)
                    model1 = MyModel(2)
                    optimizer = torch.optim.SGD([{'params': model0.
                        parameters(), 'lr': 0.25}], momentum=0.125)
                    _amp_state.allow_incoming_model_not_fp32 = True
                    [model0, model1], optimizer = amp.initialize([model0,
                        model1], optimizer, opt_level=opt_level, verbosity=
                        0, cast_model_type=False)
                    _amp_state.allow_incoming_model_not_fp32 = False
                    _amp_state.loss_scalers[0]._loss_scale = 4.0
                    self.zero_grad([model0, model1], optimizer, how_to_zero)
                    loss = model0(self.x)
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    optimizer.step()
                    if zero_before_add:
                        self.zero_grad([model0, model1], optimizer, how_to_zero
                            )
                    optimizer.add_param_group({'params': model1.parameters(
                        ), 'lr': 0.5})
                    if not zero_before_add:
                        self.zero_grad([model0, model1], optimizer, how_to_zero
                            )
                    loss = model0(self.x) + model1(self.x)
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward(retain_graph=try_accumulation)
                    if try_accumulation:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    optimizer.step()
                    self.zero_grad([model0, model1], optimizer, how_to_zero)
                    loss = model0(self.x) + model1(self.x)
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward(retain_graph=try_accumulation)
                    if try_accumulation:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    optimizer.step()
                    final_params = [param.data.clone() for param in model0.
                        parameters()] + [param.data.clone() for param in
                        model1.parameters()]
                    for reference, final in zip(reference_params, final_params
                        ):
                        self.assertTrue(torch.allclose(reference.to(final.
                            dtype), final),
                            'opt_level = {}, how_to_zero = {}, zero_before_add = {}'
                            .format(opt_level, how_to_zero, zero_before_add))

@unittest.skipIf(disabled, 'amp_C is unavailable')
def test_2models2losses2optimizers(self):
    model0 = MyModel(1)
    model1 = MyModel(2)
    optimizer0 = torch.optim.SGD([{'params': model0.parameters(), 'lr': 
        0.25}], momentum=0.125)
    optimizer1 = torch.optim.SGD([{'params': model1.parameters(), 'lr': 0.5
        }], momentum=0.25)
    reference_grads = [[], [], [], [], []]
    final_params = [None, None, None, None, None]
    for i in range(2):
        optimizer0.zero_grad()
        optimizer1.zero_grad()
        loss0 = model0(self.x)
        loss1 = model1(self.x)
        loss0.backward()
        loss1.backward()
        reference_grads[0].append([param.grad.data.clone() for param in
            model0.parameters()] + [param.grad.data.clone() for param in
            model1.parameters()])
        optimizer0.step()
        optimizer1.step()
    final_params[0] = [param.data.clone() for param in model0.parameters()] + [
        param.data.clone() for param in model1.parameters()]

    def what_got_skipped(which_iter, which_backward):
        if which_iter == 0 and which_backward == 0:
            return 1
        if which_iter == 0 and which_backward == 1:
            return 2
        if which_iter == 1 and which_backward == 0:
            return 3
        if which_iter == 1 and which_backward == 1:
            return 4
        return 0
    for which_iter in (0, 1):
        for which_backward in (0, 1):
            model0 = MyModel(1)
            model1 = MyModel(2)
            optimizer0 = torch.optim.SGD([{'params': model0.parameters(),
                'lr': 0.25}], momentum=0.125)
            optimizer1 = torch.optim.SGD([{'params': model1.parameters(),
                'lr': 0.5}], momentum=0.25)
            for i in range(3):
                optimizer0.zero_grad()
                optimizer1.zero_grad()
                loss0 = model0(self.x)
                loss1 = model1(self.x)
                loss0.backward()
                loss1.backward()
                if i != which_iter:
                    reference_grads[what_got_skipped(which_iter,
                        which_backward)].append([param.grad.data.clone() for
                        param in model0.parameters()] + [param.grad.data.
                        clone() for param in model1.parameters()])
                if i == which_iter:
                    if which_backward == 0:
                        optimizer1.step()
                    else:
                        optimizer0.step()
                else:
                    optimizer0.step()
                    optimizer1.step()
            final_params[what_got_skipped(which_iter, which_backward)] = [param
                .data.clone() for param in model0.parameters()] + [param.
                data.clone() for param in model1.parameters()]
    for materialize_master_grads in (False, True):
        for opt_level in ('O0', 'O1', 'O2', 'O3'):
            for how_to_zero in ('none', 'model', 'optimizer'):
                for use_multiple_loss_scalers in (False, True):
                    if opt_level == 'O1' or opt_level == 'O2':
                        inject_inf_iters = -1, 0, 1
                    else:
                        inject_inf_iters = -1,
                    for inject_inf in inject_inf_iters:
                        if inject_inf >= 0:
                            inject_inf_locs = 'fp16', 'fp32'
                            which_backwards = 0, 1
                        else:
                            inject_inf_locs = 'fdsa',
                            which_backwards = None,
                        for inject_inf_loc in inject_inf_locs:
                            for which_backward in which_backwards:
                                if use_multiple_loss_scalers:
                                    num_losses = 2
                                    loss_ids = [0, 1]
                                else:
                                    num_losses = 1
                                    loss_ids = [0, 0]
                                if inject_inf >= 0:
                                    iters = 3
                                else:
                                    iters = 2
                                model0 = MyModel(1)
                                model1 = MyModel(2)
                                models = [model0, model1]
                                optimizer0 = FusedSGD([{'params': model0.
                                    parameters(), 'lr': 0.25}], momentum=
                                    0.125, materialize_master_grads=
                                    materialize_master_grads)
                                optimizer1 = FusedSGD([{'params': model1.
                                    parameters(), 'lr': 0.5}], momentum=
                                    0.25, materialize_master_grads=
                                    materialize_master_grads)
                                _amp_state.allow_incoming_model_not_fp32 = True
                                [model0, model1], [optimizer0, optimizer1
                                    ] = amp.initialize([model0, model1], [
                                    optimizer0, optimizer1], opt_level=
                                    opt_level, verbosity=0, cast_model_type
                                    =False, num_losses=num_losses)
                                (_amp_state.allow_incoming_model_not_fp32
                                    ) = False
                                _amp_state.loss_scalers[0]._loss_scale = 4.0
                                if use_multiple_loss_scalers:
                                    _amp_state.loss_scalers[1
                                        ]._loss_scale = 16.0
                                unskipped = 0
                                for i in range(iters):
                                    if how_to_zero == 'none':
                                        for model in models:
                                            for param in model.parameters():
                                                param.grad = None
                                    elif how_to_zero == 'model':
                                        for model in models:
                                            model.zero_grad()
                                    else:
                                        optimizer0.zero_grad()
                                        optimizer1.zero_grad()
                                    loss0 = model0(self.x)
                                    loss1 = model1(self.x)
                                    with amp.scale_loss(loss0, optimizer0,
                                        loss_id=loss_ids[0]) as scaled_loss:
                                        scaled_loss.backward()
                                        if (i == inject_inf and which_backward == 0
                                            ):
                                            if inject_inf_loc == 'fp32':
                                                model0.weight0.grad[0] = float('inf')
                                            elif inject_inf_loc == 'fp16':
                                                model0.weight1.grad[0] = float('inf')
                                    with amp.scale_loss(loss1, optimizer1,
                                        loss_id=loss_ids[1]) as scaled_loss:
                                        scaled_loss.backward()
                                        if (i == inject_inf and which_backward == 1
                                            ):
                                            if inject_inf_loc == 'fp32':
                                                model1.weight0.grad[0] = float('inf')
                                            elif inject_inf_loc == 'fp16':
                                                model1.weight1.grad[0] = float('inf')
                                    if i != inject_inf:
                                        master_params = list(amp.master_params(
                                            optimizer0)) + list(amp.
                                            master_params(optimizer1))
                                        for param, reference_grad in zip(
                                            master_params, reference_grads[
                                            what_got_skipped(inject_inf,
                                            which_backward)][unskipped]):
                                            if (opt_level == 'O2' and not
                                                materialize_master_grads):
                                                continue
                                            else:
                                                self.assertTrue(torch.allclose(param.
                                                    grad.float(), reference_grad.float()))
                                        unskipped += 1
                                    optimizer0.step()
                                    optimizer1.step()
                                model_params = [p for p in model0.parameters()
                                    ] + [p for p in model1.parameters()]
                                master_params = [p for p in amp.
                                    master_params(optimizer0)] + [p for p in
                                    amp.master_params(optimizer1)]
                                for model, master, reference in zip(
                                    model_params, master_params,
                                    final_params[what_got_skipped(
                                    inject_inf, which_backward)]):
                                    self.assertTrue(torch.allclose(model,
                                        reference))
                                    self.assertTrue(torch.allclose(model,
                                        master.to(model.dtype)))
                                if opt_level == 'O1':
                                    _amp_state.handle._deactivate()

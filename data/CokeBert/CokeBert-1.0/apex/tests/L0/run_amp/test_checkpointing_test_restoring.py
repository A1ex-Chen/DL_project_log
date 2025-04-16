def test_restoring(self):
    nb_epochs = 10
    nb_epochs_restore = nb_epochs // 2
    for opt_level in self.test_opt_levels:
        for res_opt_level in self.test_opt_levels:
            for amp_before_load in [True, False]:
                for num_losses in range(1, 3):
                    test_setup = ('#' * 75 + '\n' +
                        f'opt_level {opt_level}\n' +
                        f"""restore_opt_level {res_opt_level}
""" +
                        f'amp_before_load {amp_before_load}\n' +
                        f'num_losses {num_losses}\n')
                    self.seed()
                    model = MyModel().to('cuda')
                    optimizer = optim.SGD(model.parameters(), lr=self.
                        initial_lr)
                    model, optimizer = amp.initialize(model, optimizer,
                        opt_level=opt_level, num_losses=num_losses * 2,
                        verbosity=0)
                    if opt_level == res_opt_level:
                        for epoch in range(nb_epochs):
                            x = torch.randn(16, 3, 24, 24, device='cuda')
                            output = self.train_step(model, optimizer, x,
                                range(num_losses))
                            if epoch == nb_epochs_restore - 1:
                                checkpoint = {'model': model.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'amp': amp.state_dict()}
                                self.check_state_dict_fp32(checkpoint['model'])
                                restore_model = MyModel().to('cuda')
                                restore_optimizer = optim.SGD(restore_model
                                    .parameters(), lr=self.initial_lr)
                                if amp_before_load:
                                    restore_model, restore_optimizer = (amp
                                        .initialize(restore_model,
                                        restore_optimizer, opt_level=
                                        res_opt_level, num_losses=
                                        num_losses * 2, verbosity=0))
                                restore_model.load_state_dict(checkpoint[
                                    'model'])
                                restore_optimizer.load_state_dict(checkpoint
                                    ['optimizer'])
                                if not amp_before_load:
                                    restore_model, restore_optimizer = (amp
                                        .initialize(restore_model,
                                        restore_optimizer, opt_level=
                                        res_opt_level, num_losses=
                                        num_losses * 2, verbosity=0))
                            elif epoch >= nb_epochs_restore:
                                restore_output = self.train_step(restore_model,
                                    restore_optimizer, x, range(num_losses,
                                    num_losses * 2))
                                self.assertTrue(torch.allclose(output.float
                                    (), restore_output.float()), 
                                    'Output of reference and restored models differ for '
                                     + test_setup)
                                self.compare_models(model, restore_model,
                                    test_setup)
                    else:
                        continue

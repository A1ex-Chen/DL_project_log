def test_state_dict(self):
    for opt_level in self.test_opt_levels:
        if opt_level == 'O3':
            continue
        model = MyModel().to('cuda')
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model, optimizer = amp.initialize(model, optimizer, opt_level=
            opt_level, verbosity=0)
        state_dict = model.state_dict()
        for key in state_dict:
            self.assertFalse('Half' in state_dict[key].type())
        data = torch.randn(10, 3, 4, 4, device='cuda')
        target = torch.randn(10, 6, 4, 4, device='cuda')
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        last_loss = loss.item()
        for epoch in range(10):
            optimizer.zero_grad()
            output = model(data)
            loss = F.mse_loss(output, target)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            self.assertTrue(loss.item() < last_loss)
            last_loss = loss.item()

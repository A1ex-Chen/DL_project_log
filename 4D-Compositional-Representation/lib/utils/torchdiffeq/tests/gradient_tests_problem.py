def problem(self):


    class Odefunc(torch.nn.Module):

        def __init__(self):
            super(Odefunc, self).__init__()
            self.A = torch.nn.Parameter(torch.tensor([[-0.1, 2.0], [-2.0, -
                0.1]]))
            self.unused_module = torch.nn.Linear(2, 5)

        def forward(self, t, y):
            return torch.mm(y ** 3, self.A)
    y0 = torch.tensor([[2.0, 0.0]]).to(TEST_DEVICE).requires_grad_(True)
    t_points = torch.linspace(0.0, 25.0, 10).to(TEST_DEVICE).requires_grad_(
        True)
    func = Odefunc().to(TEST_DEVICE)
    return func, y0, t_points

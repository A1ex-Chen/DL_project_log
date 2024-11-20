def _grid_constructor_from_step_size(self, step_size):

    def _grid_constructor(func, y0, t):
        start_time = t[0]
        end_time = t[-1]
        niters = torch.ceil((end_time - start_time) / step_size + 1).item()
        t_infer = torch.arange(0, niters).to(t) * step_size + start_time
        """
            if t_infer[-1] > t[-1]:
                t_infer[-1] = t[-1]
            """
        if t_infer[-1] != t[-1]:
            t_infer[-1] = t[-1]
        return t_infer
    return _grid_constructor

def get_order_list(self, num_inference_steps: int) ->List[int]:
    """
        Computes the solver order at each time step.

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
        """
    steps = num_inference_steps
    order = self.config.solver_order
    if order > 3:
        raise ValueError('Order > 3 is not supported by this scheduler')
    if self.config.lower_order_final:
        if order == 3:
            if steps % 3 == 0:
                orders = [1, 2, 3] * (steps // 3 - 1) + [1, 2] + [1]
            elif steps % 3 == 1:
                orders = [1, 2, 3] * (steps // 3) + [1]
            else:
                orders = [1, 2, 3] * (steps // 3) + [1, 2]
        elif order == 2:
            if steps % 2 == 0:
                orders = [1, 2] * (steps // 2 - 1) + [1, 1]
            else:
                orders = [1, 2] * (steps // 2) + [1]
        elif order == 1:
            orders = [1] * steps
    elif order == 3:
        orders = [1, 2, 3] * (steps // 3)
    elif order == 2:
        orders = [1, 2] * (steps // 2)
    elif order == 1:
        orders = [1] * steps
    return orders

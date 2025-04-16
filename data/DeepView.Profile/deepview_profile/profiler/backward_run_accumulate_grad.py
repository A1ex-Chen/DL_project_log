def run_accumulate_grad(self):
    for grad_fn, grad in self._ag_dict.items():
        grad_fn(grad)

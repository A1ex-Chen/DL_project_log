def synchronize_between_processes(self):
    """
        Warning: does not synchronize the deque!
        """
    if not is_distributed():
        return
    t = torch.tensor([self.count, self.total], dtype=torch.float64, device=
        'cuda')
    barrier()
    all_reduce_sum(t)
    t = t.tolist()
    self.count = int(t[0])
    self.total = t[1]

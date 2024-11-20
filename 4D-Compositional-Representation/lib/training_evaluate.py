def evaluate(self, val_loader):
    """ Performs an evaluation.
        Args:
            val_loader (dataloader): Pytorch dataloader
        """
    eval_list = defaultdict(list)
    for data in tqdm(val_loader):
        eval_step_dict = self.eval_step(data)
        for k, v in eval_step_dict.items():
            eval_list[k].append(v)
    eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
    return eval_dict

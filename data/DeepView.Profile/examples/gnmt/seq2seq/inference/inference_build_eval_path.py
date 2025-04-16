def build_eval_path(self, epoch, iteration):
    """
        Appends index of the current epoch and index of the current iteration
        to the name of the file with results.

        :param epoch: index of the current epoch
        :param iteration: index of the current iteration
        """
    if iteration is not None:
        eval_fname = f'eval_epoch_{epoch}_iter_{iteration}'
    else:
        eval_fname = f'eval_epoch_{epoch}'
    eval_path = os.path.join(self.save_path, eval_fname)
    return eval_path

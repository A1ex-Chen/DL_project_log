def after_train(self):
    if (self._eval_after_train and self.trainer.iter + 1 >= self.trainer.
        max_iter):
        self._do_eval()
    del self._func

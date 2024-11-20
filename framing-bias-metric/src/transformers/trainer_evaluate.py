def evaluate(self, eval_dataset: Optional[Dataset]=None, ignore_keys:
    Optional[List[str]]=None) ->Dict[str, float]:
    """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
    if eval_dataset is not None and not isinstance(eval_dataset,
        collections.abc.Sized):
        raise ValueError('eval_dataset must implement __len__')
    eval_dataloader = self.get_eval_dataloader(eval_dataset)
    output = self.prediction_loop(eval_dataloader, description='Evaluation',
        prediction_loss_only=True if self.compute_metrics is None else None,
        ignore_keys=ignore_keys)
    self.log(output.metrics)
    if self.args.tpu_metrics_debug or self.args.debug:
        xm.master_print(met.metrics_report())
    self.control = self.callback_handler.on_evaluate(self.args, self.state,
        self.control, output.metrics)
    return output.metrics

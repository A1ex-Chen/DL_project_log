def get_eval_dataloader(self, eval_dataset: Optional[Dataset]=None
    ) ->DataLoader:
    """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                If provided, will override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`, columns not
                accepted by the ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
    if eval_dataset is None and self.eval_dataset is None:
        raise ValueError('Trainer: evaluation requires an eval_dataset.')
    elif eval_dataset is not None and not isinstance(eval_dataset,
        collections.abc.Sized):
        raise ValueError('eval_dataset must implement __len__')
    elif is_datasets_available() and isinstance(eval_dataset, datasets.Dataset
        ):
        self._remove_unused_columns(eval_dataset, description='evaluation')
    eval_dataset = (eval_dataset if eval_dataset is not None else self.
        eval_dataset)
    eval_sampler = self._get_eval_sampler(eval_dataset)
    return DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.
        args.eval_batch_size, collate_fn=self.data_collator, drop_last=self
        .args.dataloader_drop_last, num_workers=self.args.
        dataloader_num_workers)

def get_test_dataloader(self, test_dataset: Dataset) ->DataLoader:
    """
        Returns the test :class:`~torch.utils.data.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                The test dataset to use. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
    if not isinstance(test_dataset, collections.abc.Sized):
        raise ValueError('test_dataset must implement __len__')
    elif is_datasets_available() and isinstance(test_dataset, datasets.Dataset
        ):
        self._remove_unused_columns(test_dataset, description='test')
    test_sampler = self._get_eval_sampler(test_dataset)
    return DataLoader(test_dataset, sampler=test_sampler, batch_size=self.
        args.eval_batch_size, collate_fn=self.data_collator, drop_last=self
        .args.dataloader_drop_last)

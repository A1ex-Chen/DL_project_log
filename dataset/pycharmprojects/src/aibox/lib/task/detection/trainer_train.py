def train(self):
    num_batches_in_epoch = len(self._dataloader)
    self._callback.on_train_begin(num_batches_in_epoch)
    for epoch in self._epoch_range:
        self._callback.on_epoch_begin(epoch, num_batches_in_epoch)
        for batch_index, item_tuple_batch in enumerate(self._dataloader):
            n_batch = batch_index + 1
            self._callback.on_batch_begin(epoch, num_batches_in_epoch, n_batch)
            item_batch = [Dataset.Item(*item_tuple) for item_tuple in
                item_tuple_batch]
            processed_image_batch = Bunch([it.processed_image for it in
                item_batch])
            processed_bboxes_batch = Bunch([it.processed_bboxes for it in
                item_batch])
            classes_batch = Bunch([it.classes for it in item_batch])
            (anchor_objectness_loss_batch, anchor_transformer_loss_batch,
                proposal_class_loss_batch, proposal_transformer_loss_batch) = (
                self._model.forward(processed_image_batch,
                processed_bboxes_batch, classes_batch))
            anchor_objectness_loss = torch.stack(anchor_objectness_loss_batch,
                dim=0).mean()
            anchor_transformer_loss = torch.stack(anchor_transformer_loss_batch
                , dim=0).mean()
            proposal_class_loss = torch.stack(proposal_class_loss_batch, dim=0
                ).mean()
            proposal_transformer_loss = torch.stack(
                proposal_transformer_loss_batch, dim=0).mean()
            loss = (anchor_objectness_loss + anchor_transformer_loss +
                proposal_class_loss + proposal_transformer_loss)
            assert not torch.isnan(loss).any(
                ), 'Got `nan` loss. Please reduce the learning rate and try again.'
            if epoch == 1 and batch_index == 0:
                self._callback.save_model_graph(image_shape=
                    processed_image_batch[0].shape)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            self._scheduler.warm_step()
            self._callback.on_batch_end(epoch, num_batches_in_epoch,
                n_batch, loss.item(), anchor_objectness_loss.item(),
                anchor_transformer_loss.item(), proposal_class_loss.item(),
                proposal_transformer_loss.item())
            if self._callback.is_terminated():
                break
        self._scheduler.step()
        if self._callback.is_terminated():
            break
        self._callback.on_epoch_end(epoch, num_batches_in_epoch)
    self._callback.on_train_end(num_batches_in_epoch)
    for datasets in [self.train_dataset.datasets, self.val_dataset.datasets,
        self.test_dataset.datasets]:
        for dataset in datasets:
            dataset: Dataset
            dataset.teardown_lmdb()

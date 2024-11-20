def resize_num_qa_labels(self, num_labels):
    """
        Build a resized question answering linear layer Module from a provided new linear layer. Increasing the size
        will add newly initialized weights. Reducing the size will remove weights from the end

        Args:
            num_labels (:obj:`int`, `optional`):
                New number of labels in the linear layer weight matrix. Increasing the size will add newly initialized
                weights at the end. Reducing the size will remove weights from the end. If not provided or :obj:`None`,
                just returns a pointer to the qa labels :obj:`torch.nn.Linear`` module of the model without doing
                anything.

        Return:
            :obj:`torch.nn.Linear`: Pointer to the resized Linear layer or the old Linear layer
        """
    cur_qa_logit_layer = self.get_qa_logit_layer()
    if num_labels is None or cur_qa_logit_layer is None:
        return
    new_qa_logit_layer = self._resize_qa_labels(num_labels)
    self.config.num_qa_labels = num_labels
    self.num_qa_labels = num_labels
    return new_qa_logit_layer

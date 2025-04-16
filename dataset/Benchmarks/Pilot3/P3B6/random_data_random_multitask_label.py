def random_multitask_label(self, num_classes):
    return torch.FloatTensor(num_classes).random_(0, 2)

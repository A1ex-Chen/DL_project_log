def dummy_sample(self, num_vec_classes):
    batch_size = 4
    height = 8
    width = 8
    sample = torch.randint(0, num_vec_classes, (batch_size, height * width))
    return sample

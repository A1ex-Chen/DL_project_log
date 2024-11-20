def prepare_class_labels(self, batch_size, device, class_labels=None):
    if self.unet.config.num_class_embeds is not None:
        if isinstance(class_labels, list):
            class_labels = torch.tensor(class_labels, dtype=torch.int)
        elif isinstance(class_labels, int):
            assert batch_size == 1, 'Batch size must be 1 if classes is an int'
            class_labels = torch.tensor([class_labels], dtype=torch.int)
        elif class_labels is None:
            class_labels = torch.randint(0, self.unet.config.
                num_class_embeds, size=(batch_size,))
        class_labels = class_labels.to(device)
    else:
        class_labels = None
    return class_labels

def create_labels(self, num_classes, num_docs):
    labels = [self.random_multitask_label(num_classes) for _ in range(num_docs)
        ]
    return torch.stack(labels)

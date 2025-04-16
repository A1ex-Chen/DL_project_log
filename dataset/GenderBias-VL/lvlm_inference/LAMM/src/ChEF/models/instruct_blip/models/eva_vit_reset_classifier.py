def reset_classifier(self, num_classes, global_pool=''):
    self.num_classes = num_classes
    self.head = nn.Linear(self.embed_dim, num_classes
        ) if num_classes > 0 else nn.Identity()

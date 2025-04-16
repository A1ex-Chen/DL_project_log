def reset_classifier(self, num_classes: int, global_pool=None) ->None:
    self.num_classes = num_classes
    if global_pool is not None:
        assert global_pool in ('', 'avg', 'token', 'map')
        if global_pool == 'map' and self.attn_pool is None:
            assert False, 'Cannot currently add attention pooling in reset_classifier().'
        elif global_pool != 'map ' and self.attn_pool is not None:
            self.attn_pool = None
        self.global_pool = global_pool
    self.head = nn.Linear(self.embed_dim, num_classes
        ) if num_classes > 0 else nn.Identity()

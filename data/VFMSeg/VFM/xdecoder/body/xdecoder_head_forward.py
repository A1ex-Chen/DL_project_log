def forward(self, features, mask=None, target_queries=None, target_vlp=None,
    task='seg', extra={}):
    return self.layers(features, mask, target_queries, target_vlp, task, extra)

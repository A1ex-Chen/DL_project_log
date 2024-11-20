def _add_instance_ids(self, key='instance_id'):
    for idx, ann in enumerate(self.annotation):
        ann[key] = str(idx)

def update(self, results):
    """
        Args:
            results: Dict{'boxes': Dict{}, 'masks': Dict{}}
        """
    self.metric_box.update(list(results['boxes'].values()))
    self.metric_mask.update(list(results['masks'].values()))

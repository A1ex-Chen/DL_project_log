def _rebuild_predictions_with_mask(self, mask: List[bool], predictions:
    Dict[str, 'np.ndarray']) ->List[Dict[str, Any]]:
    """Merge the just-computed predictions with empty vectors for empty input items.
        Making sure everything is well-aligned"
        """
    i = 0
    results = []
    for mask_value in mask:
        if mask_value:
            results.append(self._generate_empty_prediction())
        else:
            results.append({name: predictions[name][i] for name in self.
                output_tensor_mapping})
            i += 1
    return results

def _transpose_dense_predictions(self, predictions: List[List[Tensor]],
    dims_per_anchor: List[int]) ->List[List[Tensor]]:
    """
        Transpose the dense per-level predictions.

        Args:
            predictions: a list of outputs, each is a list of per-level
                predictions with shape (N, Ai x K, Hi, Wi), where N is the
                number of images, Ai is the number of anchors per location on
                level i, K is the dimension of predictions per anchor.
            dims_per_anchor: the value of K for each predictions. e.g. 4 for
                box prediction, #classes for classification prediction.

        Returns:
            List[List[Tensor]]: each prediction is transposed to (N, Hi x Wi x Ai, K).
        """
    assert len(predictions) == len(dims_per_anchor)
    res: List[List[Tensor]] = []
    for pred, dim_per_anchor in zip(predictions, dims_per_anchor):
        pred = [permute_to_N_HWA_K(x, dim_per_anchor) for x in pred]
        res.append(pred)
    return res

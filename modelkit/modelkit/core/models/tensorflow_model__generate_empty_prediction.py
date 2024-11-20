def _generate_empty_prediction(self) ->Dict[str, Any]:
    """Function used to fill in values when rebuilding predictions with the mask"""
    return {name: np.zeros((1,) + self.output_shapes[name], self.
        output_dtypes[name]) for name in self.output_tensor_mapping}

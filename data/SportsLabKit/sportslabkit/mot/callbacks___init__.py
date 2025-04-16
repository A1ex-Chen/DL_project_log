def __init__(self, vector_model: BaseVectorModel):
    """Initialize TeamClassificationCallback.

        Args:
            vector_model (BaseVectorModel): A trained object responsible for classifying teams.
                            This object is generally loaded from a pickle file that
                            contains a trained scikit-learn Pipeline.
                - The object should have a `predict` method with the following specifications:
                    - predict(input_features: np.ndarray) -> np.ndarray
                        - Input: `input_features` is an ndarray of shape `(num_samples, num_features)`.
                                `num_samples` is the number of samples, and `num_features` is the
                                feature dimension for each sample.
                        - Output: An ndarray of shape `(num_samples,)` containing the predicted team IDs.
                                For a 2-class problem, it will contain integers like 0 or 1.
                - Example: If you're using an SVM-based classifier saved using pickle, this `predict`
                        method would take a feature vector and output the corresponding team IDs
                        (either 0 or 1 in a 2-class problem).

        Note:
            The `vector_model` is expected to be a serialized object (e.g., pickle file)
            conforming to the above `predict` method specifications. It's commonly generated
            using scikit-learn and saved for future use.
        """
    super().__init__()
    self.vector_model = vector_model

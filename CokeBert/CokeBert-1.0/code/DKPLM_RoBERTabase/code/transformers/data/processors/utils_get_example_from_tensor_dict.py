def get_example_from_tensor_dict(self, tensor_dict):
    """Gets an example from a dict with tensorflow tensors

        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
    raise NotImplementedError()

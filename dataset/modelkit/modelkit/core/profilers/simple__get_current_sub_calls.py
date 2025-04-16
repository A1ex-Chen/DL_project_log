def _get_current_sub_calls(self, model_name: str) ->Dict[str, int]:
    """Get the number of current sub model calls.
        Args:
            model_name (str)
        Returns:
            Dict[str, int]: sub model calls
        """
    current_calls: Dict[str, int] = {}
    for sub_model in self._get_all_subs(model_name):
        current_calls[sub_model] = self.graph_calls[sub_model]['__main__']
    return current_calls

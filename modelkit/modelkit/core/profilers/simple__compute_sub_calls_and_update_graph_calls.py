def _compute_sub_calls_and_update_graph_calls(self, model_name: str,
    previous_calls: Dict[str, int]) ->Dict[str, int]:
    """Infer all sub models call (`sub_calls`) using `previous_calls` and update
        `graph_calls` and the end of context manager.

        P.S With the 'shared' context manager, we can't directly record `sub_calls`,
            but only the current model call (incremented in
            self.graph_calls[model_name]["__main__"]).
            Using the counts in "__main__" of all models, we deduce all sub models
            calls.

        Args:
            model_name (str): current model name
            previous_calls (Dict[str, int]):

        Returns:
            Dict[str, int]: sub_calls
        """
    self.graph_calls[model_name]['__main__'] += 1
    sub_calls: Dict[str, int] = {}
    for sub_model in self._get_all_subs(model_name):
        sub_calls[sub_model] = self.graph_calls[sub_model]['__main__'
            ] - previous_calls.get(sub_model, 0)
        self.graph_calls[model_name][sub_model] += sub_calls[sub_model]
    return sub_calls

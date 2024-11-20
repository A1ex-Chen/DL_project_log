def _get_all_subs(self, model_name: str) ->Set[str]:
    """Get the set of all sub model names."""
    res = set()
    for key in self.graph[model_name]:
        if key != '__main__':
            res.add(key)
        res = res.union(self._get_all_subs(key))
    return res

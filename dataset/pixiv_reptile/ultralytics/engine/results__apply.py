def _apply(self, fn, *args, **kwargs):
    """
        Applies a function to all non-empty attributes and returns a new Results object with modified attributes. This
        function is internally called by methods like .to(), .cuda(), .cpu(), etc.

        Args:
            fn (str): The name of the function to apply.
            *args: Variable length argument list to pass to the function.
            **kwargs: Arbitrary keyword arguments to pass to the function.

        Returns:
            (Results): A new Results object with attributes modified by the applied function.

        Example:
            ```python
            results = model("path/to/image.jpg")
            for result in results:
                result_cuda = result.cuda()
                result_cpu = result.cpu()
            ```
        """
    r = self.new()
    for k in self._keys:
        v = getattr(self, k)
        if v is not None:
            setattr(r, k, getattr(v, fn)(*args, **kwargs))
    return r

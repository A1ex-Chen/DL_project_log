def get_default_steps(self) ->int:
    """Returns default number of steps recommended for inference.

        Returns:
            `int`:
                The number of steps.
        """
    return 50 if isinstance(self.scheduler, DDIMScheduler) else 1000

def update_states(self, states: dict[str, Any], global_step: (int | None)=None
    ) ->None:
    """Update multiple states with new values.

        Args:
            states (Dict[str, Any]): Dictionary of states to be updated.
            global_step (Optional[int], optional): Global step. Defaults to None.
        """
    for name, value in states.items():
        self.update_state(name, value)

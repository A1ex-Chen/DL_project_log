def store(self, parameters: Iterable[torch.nn.Parameter]) ->None:
    """
        Args:
        Save the current parameters for restoring later.
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                temporarily stored.
        """
    self.temp_stored_params = [param.detach().cpu().clone() for param in
        parameters]
